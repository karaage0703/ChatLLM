# ChatLLM
Chat test scripts for LLM


## Setup
Git clone
```sh
$ cd && git clone https://github.com/karaage0703/ChatLLM
```

### Devcontainer(VS Code)

Setup devcontainer

[Reference(Japanese)](https://zenn.dev/karaage0703/books/80b6999d429abc8051bb/viewer/6ebae8)

### Docker compose

#### Docker build
CPU

```sh
$ docker compose up base
```

GPU

```sh
$ docker compose up gpu
```

GPU(Nvidia container)

```sh
$ docker compose up nvidia
```
#### Docker run

Check docker image name

```sh
$ docker ps
```

```sh
$ docker exec -it <image name> /bin/bash

# example
# $ docker exec -it chatllm-nvidia-1 /bin/bash
```

#### Run app

```sh
root@hostname:/# cd /root
root@hostname:~# python3 chat_calm.py
root@hostname:~# python3 chat_rinna.py
root@hostname:~# python3 chat_rwkv.py
root@hostname:~# python3 chat_llama2.py
root@hostname:~# python3 chat_weblab.py
root@hostname:~# python3 chat_elyza.py
```

### Docker

#### Docker build

```
$ cd ~/ChatLLM
$ docker build -t ubuntu:ChatLLM .
```

#### Run docker
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

#### Run app

```sh
root@hostname:~# python3 chat_calm.py
root@hostname:~# python3 chat_rinna.py
root@hostname:~# python3 chat_rwkv.py
root@hostname:~# python3 chat_llama2.py
```

for stablelm

```sh
root@hostname:/# huggingface-cli login
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
- https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
- https://note.com/npaka/n/nd05b0334cb75
- https://zenn.dev/tsuzukia/articles/f886a93fc1f2fa
- https://note.com/npaka/n/nbb94b45f47a5
- https://huggingface.co/google/gemma-7b
