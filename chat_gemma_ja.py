import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from colorama import Fore, Back, Style, init

init(autoreset=True)

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained("alfredplpl/gemma-2b-it-ja-poc", cache_dir="./")
model = AutoModelForCausalLM.from_pretrained("alfredplpl/gemma-2b-it-ja-poc", quantization_config=quantization_config, cache_dir="./")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


while True:
    input_text = input('> ')
    # inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
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
    print(Fore.YELLOW + 'GEMMA: ' + output)
