import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False, cache_dir="./")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", cache_dir="./")

if torch.cuda.is_available():
    model = model.to("cuda")

messages = []

MAX_TOKENS = 2048

def encode_prompt(messages):
    prompt = "<NL>".join(messages)
    prompt += "<NL>システム: "
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    return token_ids

def trim_messages(messages, max_tokens):
    token_ids = encode_prompt(messages)
    while len(token_ids.tolist()[0]) > max_tokens:
        messages.pop(0)
        token_ids = encode_prompt(messages)
    return messages

while True:
    input_text = input('> ')
    messages.append('ユーザー: ' + input_text)
    messages = trim_messages(messages, MAX_TOKENS)

    token_ids = encode_prompt(messages)

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=128,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    messages.append('システム: ' + output)
    output = output.replace("<NL>", "\n")
    print('リンナ: ' + output)
