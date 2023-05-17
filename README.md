# ChatLLM
Chat test scripts for LLM
## Setup

```sh
$ cd && git clone https://github.com/karaage0703/ChatCyberAgent
$ cd ~/ChatCyberAgent
$ docker build -t ubuntu:ChatLLM
```

## Usage

Use GPU

```sh
$ cd ~/ChatCyberAgent
$ docker run -it -v $(pwd):/root --gpus all ubuntu:ChatLLM
```

Use CPU 

```sh
$ cd ~/ChatCyberAgent
$ docker run -it -v $(pwd):/root ubuntu:ChatLLM
```


```sh
root@hostname:/# cd /root
root@hostname:/# python3 chat_calm.py
root@hostname:/# python3 chat_rinna.py
```
## References
- https://huggingface.co/cyberagent/open-calm-7b
- https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
