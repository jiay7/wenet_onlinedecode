from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import torchaudio.compliance.kaldi as kaldi
import torchaudio
from wenet.dataset.dataset import AudioDataset, CollateFunc
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.cmvn import load_cmvn
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.asr_model import ASRModel
from python_speech_features import logfbank
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

class WeNetDecoder:
    def __init__(self,conf_file):
        with open(conf_file, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        
        # Define online feature pipeline
        feat_config = configs.get("collate_conf", {})
        self.sr = feat_config["sample_rate"]
        self.feature_type = feat_config["feature_extraction_conf"]["feature_type"]
        self.frame_shift = feat_config["feature_extraction_conf"]["frame_shift"]
        self.frame_length = feat_config["feature_extraction_conf"]["frame_length"]
        self.using_pitch = feat_config["feature_extraction_conf"]["using_pitch"]
        self.mel_bins = feat_config["feature_extraction_conf"]["mel_bins"]

        # VAD module
        self.max_sil_frame = 50

        # Wenet model
        self.model = self.get_wenetmodel(configs)
        load_checkpoint(self.model, configs["checkpoint"])
        use_cuda = int(configs["gpu"]) >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()

        # decode mode
        self.decode_mode = configs["decode_mode"]

        # dict model
        self.dict_model,self.eos = self.get_dict(configs)

        # buffer
        self.decoding_chunk_size = 16
        self.context = self.model.encoder.embed.right_context + 1 
        self.decoding_window = self.model.encoder.embed.subsampling_rate * int(self.decoding_chunk_size - 1) + self.context
        self.chunk_size = 1600
        self.stride = int(self.model.encoder.embed.subsampling_rate*self.decoding_chunk_size)
        self.sig_buffer = np.zeros(400)
        self.dat_buffer = []
        self.max_dat_buffer_block = 5000

        # cache
        self.subsampling_cache: Optional[torch.Tensor] = None
        self.elayers_output_cache: Optional[List[torch.Tensor]] = None
        self.conformer_cnn_cache: Optional[List[torch.Tensor]] = None
        self.offset = 0
        self.required_cache_size = self.decoding_chunk_size * 50
        self.outputs = []
        self.num_decode = 0

        #online decode setting
        self.rescoring = configs["rescoring"]
        if self.rescoring:
            self.ctc_weight = float(configs["model_conf"]["ctc_weight"])
            
        if self.decode_mode == "ctc_prefix_beam_search":
            self.beam,self.time_step,self.cur_hyps = self.setting_ctc_prefix_search(configs)
        elif self.decode_mode == "ctc_greedy_search":
            self.time_step,self.cur_result = self.setting_ctc_greedy_search(configs)

    def setting_ctc_greedy_search(self,configs):
        time_step = 0
        cur_result = []
        return time_step,cur_result

    def setting_ctc_prefix_search(self,configs):
        beam = configs["beam"]
        time_step = 0
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        return beam,time_step,cur_hyps

    def get_dict(self,configs):
        char_dict = {}
        with open(configs["dict"], 'r') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                char_dict[int(arr[1])] = arr[0]
        eos = len(char_dict) - 1
        return char_dict,eos

    def get_wenetmodel(self,configs):
        if configs['cmvn_file'] is not None:
            mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
            global_cmvn = GlobalCMVN(
                torch.from_numpy(mean).float(),
                torch.from_numpy(istd).float())
        else:
            global_cmvn = None
        input_dim = configs['input_dim']
        vocab_size = configs['output_dim']
        encoder_type = configs.get('encoder', 'conformer')
        if encoder_type == 'conformer':
            encoder = ConformerEncoder(input_dim,
                                    global_cmvn=global_cmvn,
                                    **configs['encoder_conf'])
        else:
            encoder = TransformerEncoder(input_dim,
                                        global_cmvn=global_cmvn,
                                        **configs['encoder_conf'])

        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                    **configs['decoder_conf'])
        ctc = CTC(vocab_size, encoder.output_size())
        model = ASRModel(
                vocab_size=vocab_size,
                encoder=encoder,
                decoder=decoder,
                ctc=ctc,
                **configs['model_conf'],
                )
        return model

    def _extract_feature(self,waveform):
        waveform = torch.from_numpy(np.expand_dims(waveform,0))
        mat = kaldi.fbank(
                    waveform,
                    num_mel_bins=self.mel_bins,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    dither=0,
                    energy_floor=0.0,
                    sample_frequency=self.sr
                )
        # mat = logfbank(waveform,nfilt=80)
        mat = mat.detach().numpy()
        return mat

    def extract_feat(self, cur_signal, pre_signal):
        """ extract acoustic features """
        if pre_signal is not None:
            signal = np.concatenate([pre_signal, cur_signal])
            rest_points = int((len(signal) - (0.015 * self.sr)) % (0.01 * self.sr))
            run_signal = signal if rest_points == 0 else signal[:-rest_points]
            feats = self._extract_feature(run_signal).astype('float32')
            feats = feats[1:,:]
        else:
            signal = cur_signal
            rest_points = int((len(signal) - (0.015 * self.sr)) % (0.01 * self.sr))
            run_signal = signal[:-rest_points]
            feats = self._extract_feature(run_signal).astype('float32')
  
        return feats, signal

    def buffer_signal(self, signal):
        rest_points = (len(signal) - (0.015 * self.sr)) % (0.01 * self.sr)
        buf_sig_len = int((0.025 * self.sr) + rest_points) # protect frame
        return signal[-buf_sig_len:]

    def detect_process(self, signal):
        # extract acoustic features
        feat, _signal_buf = self.extract_feat(signal, self.sig_buffer) #signal和self.sig_buffer拼起来提特征，然后取第二帧及以后
        self.dat_buffer.append(feat) #self.dat_buffer存储了所有特征，以list形式，每个元素是feat矩阵
        if len(self.dat_buffer) > self.max_dat_buffer_block:
            self.dat_buffer.pop(0)
        self.sig_buffer = self.buffer_signal(_signal_buf) #把_signal_buf取最后一帧的采样点

    def reset(self):
        self.subsampling_cache = None
        self.elayers_output_cache = None
        self.conformer_cnn_cache = None
        self.outputs = []
        self.offset = 0
        self.time_step = 0
        self.cur_hyps = [(tuple(), (0.0, -float('inf')))]


    def dat_buffer_read(self,decoding_window,stride):
        if self.dat_buffer == []:
            return False,-1
        feats = np.concatenate(self.dat_buffer)
        if feats.shape[0] < decoding_window:
            return False,-1
        else:
            feats_to_predict = feats[:decoding_window]
            self.dat_buffer = [feats[stride:]]
            return True,feats_to_predict

    def ctc_greedy_search_atten(self):

        while True:
            more_data,feats_to_predict = self.dat_buffer_read(self.decoding_window,self.stride)
            if not more_data:
                break
            feats_to_predict = torch.from_numpy(np.expand_dims(feats_to_predict,0)).float()
            y, self.subsampling_cache, self.elayers_output_cache,self.conformer_cnn_cache = self.model.encoder.forward_chunk(feats_to_predict, self.offset,
                                self.required_cache_size,self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache)
            
            subsampling_cache_shape = [item.shape for item in self.subsampling_cache]
            elayers_output_cache_shape = [item.shape for item in self.elayers_output_cache]
            conformer_cnn_cache_shape = [item.shape for item in self.conformer_cnn_cache]
            self.outputs.append(y)
            ys = torch.cat(self.outputs, 1)
            self.offset += y.size(1)
            ctc_probs = self.model.ctc.log_softmax(ys)
            topk_prob, topk_index = ctc_probs.topk(1, dim=2)
            topk_index = topk_index.view(ys.shape[1],)
            hyps = topk_index.tolist()
            hyps = remove_duplicates_and_blank(hyps)
            content = [self.dict_model[index] for index in hyps]
            if ''.join(content) == '':
                continue
            print(''.join(content))
            self.num_decode += 1

    def ctc_greedy_search_purn(self):

        while True:
            more_data,feats_to_predict = self.dat_buffer_read(self.decoding_window,self.stride)
            if not more_data:
                break
            feats_to_predict = torch.from_numpy(np.expand_dims(feats_to_predict,0)).float()
            y, self.subsampling_cache, self.elayers_output_cache,self.conformer_cnn_cache = self.model.encoder.forward_chunk(feats_to_predict, self.offset,
                                self.required_cache_size,self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache)
            
            subsampling_cache_shape = [item.shape for item in self.subsampling_cache]
            elayers_output_cache_shape = [item.shape for item in self.elayers_output_cache]
            conformer_cnn_cache_shape = [item.shape for item in self.conformer_cnn_cache]
            encoder_out = y
            self.offset += y.size(1)
            ctc_probs = self.model.ctc.log_softmax(encoder_out)
            topk_prob, topk_index = ctc_probs.topk(1, dim=2)
            topk_index = topk_index.view(encoder_out.shape[1],)
            hyps = topk_index.tolist()
            hyps = remove_duplicates_and_blank(hyps)
            content = [self.dict_model[index] for index in hyps]
            if ''.join(content) == '':
                continue
            self.cur_result = self.cur_result + content
            print(''.join(self.cur_result))
            self.num_decode += 1

    def decoder_recoring(self):
        encoder_out = torch.cat(self.outputs, 1)
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
        assert len(hyps) == self.beam
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device="cpu", dtype=torch.long)
            for hyp in hyps
        ], True, -1) 
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device="cpu",
                                 dtype=torch.long)
        
        hyps_pad, _ = add_sos_eos(hyps_pad, self.model.sos, self.model.eos, -1)
        hyps_lens = hyps_lens + 1
        encoder_out = encoder_out.repeat(self.beam, 1, 1)
        encoder_mask = torch.ones(self.beam,1,encoder_out.size(1),dtype=torch.bool,device="cpu")
        decoder_out, _ = self.model.decoder(encoder_out, encoder_mask, hyps_pad,hyps_lens)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.detach().numpy()
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            score += hyp[1] * self.ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        hyps = hyps[best_index][0]
        content = [self.dict_model[index] for index in hyps]
        print("Rescoing: "+''.join(content))

    def endpoint(self):
        output_step = self.max_sil_frame // self.decoding_chunk_size
        encoder_out = torch.cat(self.outputs, 1)
        if encoder_out.shape[1] < self.max_sil_frame:
            return False
        ctc_probs = self.model.ctc.log_softmax(encoder_out)
        ctc_probs = torch.exp(ctc_probs)[0,:,0]
        is_blank = ctc_probs > 0.5
        if sum(is_blank[:self.max_sil_frame]) == self.max_sil_frame:
            print("beginging sil....")
            self.outputs = self.outputs[output_step:]
        elif sum(is_blank[-1 * self.max_sil_frame:]) == self.max_sil_frame:
            self.outputs = self.outputs[:-output_step]
            print("endpoint detect!...")
            return True
        return False

    def ctc_prefix_beam_search_purn(self):

        while True:
            more_data,feats_to_predict = self.dat_buffer_read(self.decoding_window,self.stride)
            if not more_data:
                break
            feats_to_predict = torch.from_numpy(np.expand_dims(feats_to_predict,0)).float()
            y, self.subsampling_cache, self.elayers_output_cache,self.conformer_cnn_cache = self.model.encoder.forward_chunk(feats_to_predict, self.offset,
                                self.required_cache_size,self.subsampling_cache,self.elayers_output_cache,self.conformer_cnn_cache)
            
            subsampling_cache_shape = [item.shape for item in self.subsampling_cache]
            elayers_output_cache_shape = [item.shape for item in self.elayers_output_cache]
            conformer_cnn_cache_shape = [item.shape for item in self.conformer_cnn_cache]
            self.offset += y.size(1)
            if self.rescoring:
                self.outputs.append(y)
            if self.endpoint():
                if self.rescoring:
                    self.decoder_recoring()
                self.reset()
                break

            encoder_out = y
            maxlen = encoder_out.size(1)
            ctc_probs = self.model.ctc.log_softmax(encoder_out)
            ctc_probs = ctc_probs.squeeze(0)
            for t in range(0, maxlen):
                logp = ctc_probs[t]  # (vocab_size,)
                # key: prefix, value (pb, pnb), default value(-inf, -inf)
                next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
                # 2.1 First beam prune: select topk best
                top_k_logp, top_k_index = logp.topk(self.beam)  # (beam_size,)
                for s in top_k_index:
                    s = s.item()
                    ps = logp[s].item()
                    for prefix, (pb, pnb) in self.cur_hyps:
                        last = prefix[-1] if len(prefix) > 0 else None
                        if s == 0:  # blank
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pb = log_add([n_pb, pb + ps, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                        elif s == last:
                            #  Update *ss -> *s;
                            n_pb, n_pnb = next_hyps[prefix]
                            n_pnb = log_add([n_pnb, pnb + ps])
                            next_hyps[prefix] = (n_pb, n_pnb)
                            # Update *s-s -> *ss, - is for blank
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)
                        else:
                            n_prefix = prefix + (s, )
                            n_pb, n_pnb = next_hyps[n_prefix]
                            n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                            next_hyps[n_prefix] = (n_pb, n_pnb)

                # 2.2 Second beam prune
                next_hyps = sorted(next_hyps.items(),
                                key=lambda x: log_add(list(x[1])),
                                reverse=True)
                self.cur_hyps = next_hyps[:self.beam]
            hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in self.cur_hyps]
            hyps = hyps[0][0]
            content = [self.dict_model[index] for index in hyps]
            #print("content:",content)
            if ''.join(content) == '':
                continue
            print('Partial:'+''.join(content))


    def detect(self, signal):
        chunk_num = int(len(signal) / self.chunk_size)
        for index in range(chunk_num):
            self.detect_process(signal[index * self.chunk_size : (index + 1) * self.chunk_size])
        return 0



