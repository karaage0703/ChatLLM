from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from colorama import Fore, Back, Style, init

init(autoreset=True)

model_kwargs = {"trust_remote_code": True, "device_map": "auto", "low_cpu_mem_usage": True}
model_kwargs["variant"] = "int8"
model_kwargs["load_in_8bit"] = True

tokenizer = AutoTokenizer.from_pretrained(
    "novelai/nerdstash-tokenizer-v1",
    use_fast=False, cache_dir="./",  additional_special_tokens=['▁▁']
)
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/japanese-stablelm-instruct-alpha-7b", **model_kwargs,
    cache_dir="./"
)

model.eval()

while True:
    input_text = input('> ')

    prompt = """### 指示：""" + input_text + """\n\n### 応答:"""

    input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt")
    tokens = model.generate(
        input_ids.to(device=model.device),
        max_new_tokens=256,
        temperature=1,
        top_p=0.95,
        do_sample=True,
    )

    output = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=False).strip()

    print('user:' + input_text)
    print(Fore.YELLOW + 'stablelm:' + output)
