## Conda环境配置步骤



```shell
# 1. 创建并激活Conda环境
conda create -n deepseek-r1 python=3.10
conda activate deepseek-r1
# 2. 安装PyTorch与CUDA支持
conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
#3. 安装Hugging Face相关库
pip install transformers accelerate bitsandbytes sentencepiece


pip uninstall numpy -y
pip install "numpy<2"  # 安装NumPy 1.x版本

pip install openai
pip install gradio
pip install bitsandbytes

pip install gradio_checkboxgroupmarkdown

```

一行执行

```shell
conda create -n deepseek-r1 python=3.10 -y && conda activate deepseek-r1 && conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && pip install transformers accelerate bitsandbytes sentencepiece "numpy<2" openai gradio
```

