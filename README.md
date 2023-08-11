# ChatLLM
Chat test scripts for LLM
## Setup

```sh
$ cd && git clone https://github.com/karaage0703/ChatLLM
$ cd ~/ChatLLM
$ docker build -t ubuntu:ChatLLM .
```

## Usage

### Run docker

Use GPU

```sh
$ cd ~/ChatLLM
$ docker run -it --rm -v $(pwd):/root --gpus all ubuntu:ChatLLM
```

Use CPU

```sh
$ cd ~/ChatLLM
$ docker run -it --rm -v $(pwd):/root ubuntu:ChatLLM
```

### Run app

```sh
root@hostname:/# cd /root
root@hostname:~# python3 chat_calm.py
root@hostname:~# python3 chat_rinna.py
root@hostname:~# python3 chat_rwkv.py
root@hostname:~# python3 chat_llama2.py
```

for stablelm

```sh
root@hostname:/# huggingface-cli login
root@hostname:/# cd /root
root@hostname:~# python3 chat_stablelm.py
```
## References
- https://huggingface.co/cyberagent/open-calm-7b
- https://huggingface.co/rinna/japanese-gpt-neox-3.6b-instruction-sft
- https://nowokay.hatenablog.com/entry/2023/05/17/144518
- https://note.com/npaka/n/n401dccfadedc
- https://note.com/npaka/n/n0ad63134fbe2
- https://zenn.dev/karaage0703/articles/2b753b4dc26471
- https://zenn.dev/karaage0703/articles/d58d79d8e77ab8
- https://zenn.dev/karaage0703/articles/d3893b551c68fa
- https://note.com/npaka/n/nfacbeb1ae709
- https://colab.research.google.com/github/mkshing/notebooks/blob/main/stabilityai_japanese_stablelm_alpha_7b.ipynb
