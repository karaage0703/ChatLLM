# ChatLLM
Chat test scripts for LLM
## Setup

```sh
$ cd && git clone https://github.com/karaage0703/ChatLLM
$ cd ~/ChatLLM
$ docker build -t ubuntu:ChatLLM .
```

## Usage

Use GPU

```sh
$ cd ~/ChatLLM
$ docker run -it -v $(pwd):/root --gpus all ubuntu:ChatLLM
```

Use CPU 

```sh
$ cd ~/ChatLLM
$ docker run -it -v $(pwd):/root ubuntu:ChatLLM
```


```sh
root@hostname:/# cd /root
root@hostname:~# python3 chat_calm.py
root@hostname:~# python3 chat_rinna.py
```
## References
- https://huggingface.co/cyberagent/open-calm-7b
- https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
- https://nowokay.hatenablog.com/entry/2023/05/17/144518
