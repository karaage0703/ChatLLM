from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
from colorama import Fore, Back, Style, init

init(autoreset=True)

model = RWKV(
    model="RWKV-4-World-3B-v1-20230619-ctx4096.pth",
    strategy="cuda fp16"
)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")

args = PIPELINE_ARGS(
    temperature = 1.0,
    top_p = 0.3, 
    top_k = 100, 
    alpha_frequency = 0.25, 
    alpha_presence = 0.25, 
    token_ban = [],
    token_stop = [0],
    chunk_len = 256) 

while True:
    input_text = input('> ')

    output = pipeline.generate(input_text, token_count=200, args=args)

    print(Fore.YELLOW + 'RWKV: ' + output)
