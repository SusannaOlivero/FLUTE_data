import torch
import transformers
import numpy as np
import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

MY_TOKEN = "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" # your_huggings face token
device = torch.device('cuda')

model_tag = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_tag, use_auth_token=MY_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_tag, use_auth_token=MY_TOKEN, torch_dtype=torch.float16, device_map=device)

k = 12

with open("FLUTE_data/FLUTE_train.json") as f:
    data_train = json.load(f)

data_train_metaphor = []
data_train_idiom = []
data_train_sarcasm = []
data_train_simile = []
for data in data_train:
    if data['data_type'] == 'metaphor':
        data_train_metaphor.append(data)
    elif data['data_type'] == 'idiom':
        data_train_idiom.append(data)
    elif data['data_type'] == 'sarcasm':
        data_train_sarcasm.append(data)
    elif data['data_type'] == 'simile':
        data_train_simile.append(data)
del data_train

# 1. METAPHOR
input_data_metaphor = []
input_data += random.sample(data_train_metaphor, k-1)

entailment_list = [item for item in input_data if item['label'] == 'Entailment']
contradiction_list = [item for item in input_data if item['label'] == 'Contradiction']
if not entailment_list:  # if it's empty
    entailment_data_ = [item for item in data_train_metaphor if item['label'] == 'Entailment']
    input_data_metaphor += random.sample(entailment_data_, 1)
    del entailment_data_
elif not contradiction_list:
    contradiction_data_ = [item for item in data_train_metaphor if item['label'] == 'Contradiction']
    input_data_metaphor += random.sample(contradiction_data_, 1)
    del contradiction_data_
else:
    input_data_metaphor += random.sample(data_train_metaphor, 1)

del data_train_metaphor

output_text_metaphor = ""
for item in input_data_metaphor:
    output_text_metaphor += f"\npremise: {item['premise']}\n"
    output_text_metaphor += f"hypothesis: {item['hypothesis']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text_metaphor += f"Answer: {answer}\n"
    output_text_metaphor += f"Explanation: {item['explanation']}\n"


# 2. IDIOM
input_data_idiom = []
input_data += random.sample(data_train_idiom, k-1)

entailment_list = [item for item in input_data if item['label'] == 'Entailment']
contradiction_list = [item for item in input_data if item['label'] == 'Contradiction']
if not entailment_list:  # if it's empty
    entailment_data_ = [item for item in data_train_idiom if item['label'] == 'Entailment']
    input_data_idiom += random.sample(entailment_data_, 1)
    del entailment_data_
elif not contradiction_list:
    contradiction_data_ = [item for item in data_train_idiom if item['label'] == 'Contradiction']
    input_data_idiom += random.sample(contradiction_data_, 1)
    del contradiction_data_
else:
    input_data_idiom += random.sample(data_train_idiom, 1)

del data_train_idiom

output_text_idiom = ""
for item in input_data_idiom:
    output_text_idiom += f"\npremise: {item['premise']}\n"
    output_text_idiom += f"hypothesis: {item['hypothesis']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text_idiom += f"Answer: {answer}\n"
    output_text_idiom += f"Explanation: {item['explanation']}\n"


# 3. SIMILE
input_data_simile = []
input_data += random.sample(data_train_simile, k-1)

entailment_list = [item for item in input_data if item['label'] == 'Entailment']
contradiction_list = [item for item in input_data if item['label'] == 'Contradiction']
if not entailment_list:  # if it's empty
    entailment_data_ = [item for item in data_train_simile if item['label'] == 'Entailment']
    input_data_simile += random.sample(entailment_data_, 1)
    del entailment_data_
elif not contradiction_list:
    contradiction_data_ = [item for item in data_train_simile if item['label'] == 'Contradiction']
    input_data_simile += random.sample(contradiction_data_, 1)
    del contradiction_data_
else:
    input_data_simile += random.sample(data_train_simile, 1)

del data_train_simile

output_text_simile = ""
for item in input_data_simile:
    output_text_simile += f"\npremise: {item['premise']}\n"
    output_text_simile += f"hypothesis: {item['hypothesis']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text_simile += f"Answer: {answer}\n"
    output_text_simile += f"Explanation: {item['explanation']}\n"


# 4. SARCASM
input_data_sarcasm = []
input_data += random.sample(data_train_sarcasm, k-1)

entailment_list = [item for item in input_data if item['label'] == 'Entailment']
contradiction_list = [item for item in input_data if item['label'] == 'Contradiction']
if not entailment_list:  # if it's empty
    entailment_data_ = [item for item in data_train_sarcasm if item['label'] == 'Entailment']
    input_data_sarcasm += random.sample(entailment_data_, 1)
    del entailment_data_
elif not contradiction_list:
    contradiction_data_ = [item for item in data_train_sarcasm if item['label'] == 'Contradiction']
    input_data_sarcasm += random.sample(contradiction_data_, 1)
    del contradiction_data_
else:
    input_data_sarcasm += random.sample(data_train_sarcasm, 1)

del data_train_sarcasm

output_text_sarcasm = ""
for item in input_data_sarcasm:
    output_text_sarcasm += f"\npremise: {item['premise']}\n"
    output_text_sarcasm += f"hypothesis: {item['hypothesis']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text_sarcasm += f"Answer: {answer}\n"
    output_text_sarcasm += f"Explanation: {item['explanation']}\n"


tokens_metaphor = tokenizer.tokenize(str(output_text_metaphor))
token_count_metaphor = len(tokens_metaphor)
tokens_idiom = tokenizer.tokenize(str(output_text_idiom))
token_count_idiom = len(tokens_idiom)
tokens_simile = tokenizer.tokenize(str(output_text_simile))
token_count_simile = len(tokens_simile)
tokens_sarcasm = tokenizer.tokenize(str(output_text_sarcasm))
token_count_sarcasm = len(tokens_sarcasm)
token_count = max(token_count_metaphor, token_count_idiom, token_count_simile, token_count_sarcasm)
length_max = token_count + 220

with open("FLUTE_data/FLUTE_test.json") as f:
    data_test = json.load(f)
    
data = data_test
for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Here you can find some examples of answers:\n"
    if data[i]["data_type"]=='metaphor'
        examples = output_text_metaphor
    elif data[i]["data_type"]=='idiom'
        examples = output_text_idiom
    elif data[i]["data_type"]=='simile'
        examples = output_text_simile
    elif data[i]["data_type"]=='sarcasm'
        examples = output_text_sarcasm
    prompt += examples
    request = "\npremise: "+premise+"\nhypothesis: "+hypothesis
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=False, temperature=0, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=length_max)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text = text.replace(prompt, '').strip()
    label_, explanation_ = text.split("Explanation:", 1)
    if "Entails." in label_:
            predictedlabel = "Entailment"
    elif "Contradicts." in label_:
            predictedlabel = "Contradiction"
    data[i]["predicted_label"] = predictedlabel
    explanation_ = explanation_.split("premise:")[0].lstrip().rstrip('\n')
    data[i]["model_explanation"] = explanation_

with open("p_class_t0_k12.json","w") as f:
    f.write(json.dumps(data,indent=4))

del data

data = data_test
for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Here you can find some examples of answers:\n"
    if data[i]["data_type"]=='metaphor'
        examples = output_text_metaphor
    elif data[i]["data_type"]=='idiom'
        examples = output_text_idiom
    elif data[i]["data_type"]=='simile'
        examples = output_text_simile
    elif data[i]["data_type"]=='sarcasm'
        examples = output_text_sarcasm
    prompt += examples
    request = "\npremise: "+premise+"\nhypothesis: "+hypothesis
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=True, temperature=0.3, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=length_max)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text = text.replace(prompt, '').strip()
    label_, explanation_ = text.split("Explanation:", 1)
    if "Entails." in label_:
            predictedlabel = "Entailment"
    elif "Contradicts." in label_:
            predictedlabel = "Contradiction"
    data[i]["predicted_label"] = predictedlabel
    explanation_ = explanation_.split("premise:")[0].lstrip().rstrip('\n')
    data[i]["model_explanation"] = explanation_

with open("p_class_t03_k12.json","w") as f:
    f.write(json.dumps(data,indent=4))
