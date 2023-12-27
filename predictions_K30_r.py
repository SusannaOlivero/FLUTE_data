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

with open("FLUTE_data/FLUTE_train_2.json") as f:
    data_train = json.load(f)

k = 30
method = 'random'

input_data = []

if k == 2:
    data_entail = []
    data_contra = []
    for data in data_train:
        if data['label'] == 'Entailment':
            data_entail.append(data)
        elif data['label'] == 'Contradiction':
            data_contra.append(data)
    input_data += random.sample(data_entail, 1)
    input_data += random.sample(data_contra, 1)

    del data_entail
    del data_contra

elif method == 'figure' and k > 3:
    k_type = k // 4
    k_random = k - (k_type * 4)

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

    input_data += random.sample(data_train_metaphor, k_type)
    input_data += random.sample(data_train_idiom, k_type)
    input_data += random.sample(data_train_simile, k_type)
    input_data += random.sample(data_train_sarcasm, k_type - 1)

    entailment_list = [item for item in input_data if item['label'] == 'Entailment']
    contradiction_list = [item for item in input_data if item['label'] == 'Contradiction']
    if not entailment_list:  # if it's empty
        entailment_data_sarc = [item for item in data_train_sarcasm if item['label'] == 'Entailment']
        input_data += random.sample(entailment_data_sarc, 1)
        del entailment_data_sarc
    elif not contradiction_list:
        contradiction_data_sarc = [item for item in data_train_sarcasm if item['label'] == 'Contradiction']
        input_data += random.sample(contradiction_data_sarc, 1)
        del contradiction_data_sarc
    else:
        input_data += random.sample(data_train_sarcasm, 1)

    if k_random != 0:
        i = 0
        while i < k_random:
            prova = random.sample(data_train, 1)
            if prova not in input_data:
                input_data += prova
                i += 1

    del data_train_metaphor
    del data_train_idiom
    del data_train_sarcasm
    del data_train_simile

# method = 'random'
else:
    input_data += random.sample(data_train, k - 1)

    entailment_list = [item for item in input_data if item['label'] == 'Entailment']
    contradiction_list = [item for item in input_data if item['label'] == 'Contradiction']
    if not entailment_list:
        entailment_data = [item for item in data_train if item['label'] == 'Entailment']
        input_data += random.sample(entailment_data, 1)
        del entailment_data
    elif not contradiction_list:
        contradiction_data = [item for item in data_train if item['label'] == 'Contradiction']
        input_data += random.sample(contradiction_data, 1)
        del contradiction_data
    else:
        input_data += random.sample(data_train, 1)

output_text = ""
for item in input_data:
    output_text += f"\npremise: {item['premise']}\n"
    output_text += f"hypothesis: {item['hypothesis']}\n"
    label = item['label']
    if "Entailment" in label:
        answer = "Entails."
    elif "Contradiction" in label:
        answer = "Contradicts."
    output_text += f"Answer: {answer}\n"
    output_text += f"Explanation: {item['explanation']}\n"

del data_train

examples = output_text

tokens = tokenizer.tokenize(str(examples))
token_count = len(tokens)
length_max = token_count + 150

with open("FLUTE_data/FLUTE_val_2.json") as f:
    data = json.load(f)

for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Here you can find some examples of answers:\n"
    prompt += examples
    request = "\npremise: "+premise+"\nhypothesis: "+hypothesis
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=False, temperature=0, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=length_max)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text = text.replace(prompt, '').strip()
    label_, explanation_ = text.split("Explanation:")
    if "Entails." in label_:
            predictedlabel = "Entailment"
    elif "Contradicts." in label_:
            predictedlabel = "Contradiction"
    data[i]["predicted_label"] = predictedlabel
    explanation_ = explanation_.split("premise:")[0].lstrip().rstrip('\n')
    data[i]["model_explanation"] = explanation_

with open("prediction_t0_k30_r.json","w") as f:
    f.write(json.dumps(data,indent=4))

del data

with open("FLUTE_data/FLUTE_val_2.json") as f:
    data = json.load(f)

for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Here you can find some examples of answers:\n"
    prompt += examples
    request = "\npremise: "+premise+"\nhypothesis: "+hypothesis
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=True, temperature=0.3, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=length_max)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text = text.replace(prompt, '').strip()
    label_, explanation_ = text.split("Explanation:")
    if "Entails." in label_:
            predictedlabel = "Entailment"
    elif "Contradicts." in label_:
            predictedlabel = "Contradiction"
    data[i]["predicted_label"] = predictedlabel
    explanation_ = explanation_.split("premise:")[0].lstrip().rstrip('\n')
    data[i]["model_explanation"] = explanation_

with open("prediction_t03_k30_r.json","w") as f:
    f.write(json.dumps(data,indent=4))
