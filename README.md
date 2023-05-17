# ChatCyberAgent

## Setup

```sh
$ cd && git clone https://github.com/karaage0703/ChatCyberAgent
$ cd ChatCyberAgent
$ docker build -t ubuntu:ChatCyberAgent
```

## Usage

```
$ cd ~/ChatCyberAgent
$ docker run -it -v $(pwd):/root --gpus all ubuntu:ChatCyberAgent
```

```sh
root@hostname:/# cd /root
root@hostname:/# python3 chat.py
```
