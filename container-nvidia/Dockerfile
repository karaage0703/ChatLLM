FROM nvidia/cuda:11.7.1-base-ubuntu22.04

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git python3 python3-pip

RUN pip install tokenizers>=0.13.2 prompt_toolkit numpy torch
RUN pip install rwkv transformers accelerate sentencepiece colorama
RUN pip install llama-cpp-python
RUN pip install einops bitsandbytes scipy
RUN pip install auto_gptq

WORKDIR /root
