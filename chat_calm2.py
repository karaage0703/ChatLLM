import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from colorama import Fore, Back, Style, init

init(autoreset=True)

model = AutoModelForCausalLM.from_pretrained("cyberagent/calm2-7b-chat", device_map="auto", torch_dtype="auto", cache_dir="./")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat", cache_dir='./')
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

while True:
    input_text = input('> ')
    inputs = tokenizer(input_text , return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )

    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(Fore.YELLOW + 'CALM: ' + output)