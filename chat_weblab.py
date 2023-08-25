import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import Fore, Back, Style, init

init(autoreset=True)

tokenizer = AutoTokenizer.from_pretrained(
    "matsuo-lab/weblab-10b-instruction-sft",
    cache_dir="./",
)
model = AutoModelForCausalLM.from_pretrained(
    "matsuo-lab/weblab-10b-instruction-sft",
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
    cache_dir="./",
)

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
    print(Fore.YELLOW + 'weblab:' + output)
