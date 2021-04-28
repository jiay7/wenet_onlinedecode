# wenet_onlinedeocde

## 环境配置

torch、CUDA版本以及其他python包安装，参考wenet官方文档：https://github.com/mobvoi/wenet

## 文件配置

根据exp/unified_conformer/train.yaml中的描述补全cmvn_file、checkpoint、dict等关键文件，默认模型为16k采样率数据训练所得。

## 系统运行

在正确的环境下直接执行python demo.py，默认执行offline的识别。如要修改为录音解码模式请取消demo.py文件112行注释。

