FROM amd64/ubuntu:20.04

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git python3 python3-pip

RUN pip install tokenizers>=0.13.2 prompt_toolkit numpy torch
RUN pip install rwkv transformers accelerate sentencepiece colorama
RUN pip install llama-cpp-python
