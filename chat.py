import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("cyberagent/open-calm-3b", device_map="auto", torch_dtype=torch.float16, cache_dir="./")
tokenizer = AutoTokenizer.from_pretrained("cyberagent/open-calm-3b", cache_dir='./')


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
        )
        
    output = tokenizer.decode(tokens[0], skip_special_tokens=True)
    print(output)
