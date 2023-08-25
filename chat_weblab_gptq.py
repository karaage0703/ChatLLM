from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from colorama import Fore, Back, Style, init

init(autoreset=True)

quantized_model_dir = "dahara1/weblab-10b-instruction-sft-GPTQ"
model_basename = "gptq_model-4bit-128g"

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)

model = AutoGPTQForCausalLM.from_quantized(
        quantized_model_dir,
        model_basename=model_basename,
        use_safetensors=True,
        device="cuda:0")

while True:
    input_text = input('> ')

    prompt = """### 指示：""" + input_text + """\n\n### 応答:"""

    tokens = tokenizer(prompt, return_tensors="pt").to("cuda:0").input_ids
    output = model.generate(input_ids=tokens, max_new_tokens=100, do_sample=True, temperature=0.8)

    print('user:' + input_text)
    print(Fore.YELLOW + 'weblab:' + tokenizer.decode(output[0]))
