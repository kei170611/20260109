!pip install -U "transformers>=4.51.0" accelerate -q

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

model

prompt = "大規模言語モデルで使われるTransformerの仕組みを小学生でもわかるように日本語で説明して"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Qwen3はthinking有無を切り替えができるモデル
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

model_inputs

# tokenizeされた結果を見てみる
input_ids = model_inputs["input_ids"][0]
tokens = [tokenizer.decode([tok]) for tok in input_ids.tolist()]
for tok_id, tok in zip(input_ids.tolist(), tokens):
    print(f"{tok_id:>6} : {repr(tok)}")

# トークナイザーによって元に戻されたテキスト
print(tokenizer.decode(model_inputs["input_ids"][0]))

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

generated_ids

try:
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

thinking_content

content
