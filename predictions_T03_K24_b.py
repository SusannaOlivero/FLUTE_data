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

with open("FLUTE_data/FLUTE_train.json") as f:
    data_train = json.load(f)

k = 4
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

with open("FLUTE_data/FLUTE_val.json") as f:
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
    label_ = text.split("Explanation:")[0].lstrip()
    if "Entails." in label_:
            predictedlabel = "Entailment"
    elif "Contradicts." in label_:
            predictedlabel = "Contradiction"
    explanation_ = text.split("premise:")[0].lstrip()
    explanation_ = explanation_.split("Explanation:")[1].lstrip().rstrip('\n')
    data[i]["predicted_label"] = predictedlabel
    data[i]["model_explanation"] = explanation_

with open("prediction_t03_k24_b.json","w") as f:
    f.write(json.dumps(data,indent=4))
