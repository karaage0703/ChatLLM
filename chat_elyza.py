
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from colorama import Fore, Back, Style, init

init(autoreset=True)

tokenizer = AutoTokenizer.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct",
    # "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct",
    cache_dir="./",
)
model = AutoModelForCausalLM.from_pretrained(
    "elyza/ELYZA-japanese-Llama-2-7b-instruct", 
    # "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct", 
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="./",
)

def build_prompt(text):
    prompt = """<s>[INST] <<SYS>>
    あなたは誠実で優秀な日本人のアシスタントです。
    <</SYS>>

    """ + str(text) + """[/INST]"""

    return prompt

while True:
    input_text = input('> ')

    with torch.no_grad():
        prompt = build_prompt(input_text)
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)

    print('user:' + input_text)
    print(Fore.YELLOW + 'elyza:' + output)
