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

with open("FLUTE_data/DREAM/DreamData/dream_train_SE.json") as f:
    data_train = json.load(f)

k = 1
k2 = 3*k - 1

input_data = []

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

input_data += random.sample(data_train_metaphor, k)
input_data += random.sample(data_train_idiom, k)
input_data += random.sample(data_train_simile, k)
input_data += random.sample(data_train_sarcasm, k2)

entailment_list = [item for item in input_data if item['label'] == 'Entailment']
contradiction_list = [item for item in input_data if item['label'] == 'Contradiction']
if not entailment_list: # if it's empty
    entailment_data_sarc = [item for item in data_train_sarcasm if item['label'] == 'Entailment']
    input_data += random.sample(entailment_data_sarc, 1)
    del entailment_data_sarc
elif not contradiction_list:
    contradiction_data_sarc = [item for item in data_train_sarcasm if item['label'] == 'Contradiction']
    input_data += random.sample(contradiction_data_sarc, 1)
    del contradiction_data_sarc
else:
    input_data += random.sample(data_train_sarcasm, 1)

del data_train_metaphor
del data_train_idiom
del data_train_sarcasm
del data_train_simile

del data_train

## PROMPT style 2
with open("FLUTE_data/DREAM/DreamData/dream_test_SE.json") as f:
    data_val = json.load(f)

# 1.1) Emotion
output_text = ""
for item in input_data:
    output_text += f"\npremise: {item['premise']} emotion: {item['premise-emotion']}\n"
    output_text += f"hypothesis: {item['hypothesis']} emotion: {item['hypothesis-emotion']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text += f"Answer: {answer}\n"
    output_text += f"Explanation: {item['explanation']}\n"
examples = output_text

tokens = tokenizer.tokenize(str(examples))
token_count = len(tokens)
length_max = token_count + 300

data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    prompt += "Here you can find some examples of answers:\n"
    prompt += examples
    request = "\npremise: "+data[i]["premise"]+" emotion: "+data[i]["premise-emotion"]
    request += "\nhypothesis: "+data[i]["hypothesis"]+" emotion: "+data[i]["hypothesis-emotion"]
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

with open("p_dream_k6_t0_emotion.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data

# 1.2) Motivation
output_text = ""
for item in input_data:
    output_text += f"\npremise: {item['premise']} motivation: {item['premise-motivation']}\n"
    output_text += f"hypothesis: {item['hypothesis']} motivation: {item['hypothesis-motivation']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text += f"Answer: {answer}\n"
    output_text += f"Explanation: {item['explanation']}\n"
examples = output_text

tokens = tokenizer.tokenize(str(examples))
token_count = len(tokens)
length_max = token_count + 300

data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    prompt += "Here you can find some examples of answers:\n"
    prompt += examples
    request = "\npremise: "+data[i]["premise"]+" motivation: "+data[i]["premise-motivation"]
    request += "\nhypothesis: "+data[i]["hypothesis"]+" motivation: "+data[i]["hypothesis-motivation"]
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

with open("p_dream_k6_t0_motivation.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data

# 1.3) Social Norm 
output_text = ""
for item in input_data:
    output_text += f"\npremise: {item['premise']} social-norm: {item['premise-rot']}\n"
    output_text += f"hypothesis: {item['hypothesis']} social-norm: {item['hypothesis-rot']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text += f"Answer: {answer}\n"
    output_text += f"Explanation: {item['explanation']}\n"
examples = output_text

tokens = tokenizer.tokenize(str(examples))
token_count = len(tokens)
length_max = token_count + 300

data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    prompt += "Here you can find some examples of answers:\n"
    prompt += examples
    request = "\npremise: "+data[i]["premise"]+" social-norm: "+data[i]["premise-rot"]
    request += "\nhypothesis: "+data[i]["hypothesis"]+" social-norm: "+data[i]["hypothesis-rot"]
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

with open("p_dream_k6_t0_rot.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data

# 1.4) Consequence
output_text = ""
for item in input_data:
    output_text += f"\npremise: {item['premise']} consequence: {item['premise-consequence']}\n"
    output_text += f"hypothesis: {item['hypothesis']} consequence: {item['hypothesis-consequence']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text += f"Answer: {answer}\n"
    output_text += f"Explanation: {item['explanation']}\n"
examples = output_text

tokens = tokenizer.tokenize(str(examples))
token_count = len(tokens)
length_max = token_count + 300

data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    prompt += "Here you can find some examples of answers:\n"
    prompt += examples
    request = "\npremise: "+data[i]["premise"]+" consequence: "+data[i]["premise-consequence"]
    request += "\nhypothesis: "+data[i]["hypothesis"]+" consequence: "+data[i]["hypothesis-consequence"]
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

with open("p_dream_k6_t0_consequence.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data

# 2) Scene Elaboration completed
pm = "premise-motivation"
pe = "premise-emotion"
pr = "premise-rot"
pc = "premise-consequence"
hm = "hypothesis-motivation"
he = "hypothesis-emotion"
hr = "hypothesis-rot"
hc = "hypothesis-consequence"
    
output_text = ""
for item in input_data:
    output_text += f"\npremise: {item['premise']} [motivation] {item[pm]} [emotion] {item[pe]} [social norm] {item[pr]} {item[pc]}\n"
    output_text += f"hypothesis: {item['hypothesis']} [motivation] {item[hm]} [emotion] {item[he]} [social norm] {item[hr]} {item[hc]}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text += f"Answer: {answer}\n"
    output_text += f"Explanation: {item['explanation']}\n"
examples = output_text

tokens = tokenizer.tokenize(str(examples))
token_count = len(tokens)
length_max = token_count + 350

data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    prompt += "Here you can find some examples of answers:\n"
    prompt += examples
    request = f"\npremise: {data[i]['premise']} [motivation] {data[i][pm]} [emotion] {data[i][pe]} [social norm] {data[i][pr]} {data[i][pc]}\n"
    request += f"hypothesis: {data[i]['hypothesis']} [motivation] {data[i][hm]} [emotion] {data[i][he]} [social norm] {data[i][hr]} {data[i][hc]}\n"
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

with open("p_dream_k6_t0_SE.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data
