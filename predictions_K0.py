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


with open("FLUTE_data/FLUTE_val_2.json") as f:
    data = json.load(f)

for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\nThe output must strictly follow the following format: Label: 'Entails' or 'Contradicts', Explanation: 'text'"
    request = "\npremise: "+premise+"\nhypothesis: "+hypothesis
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=False, temperature=0, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=256)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text = text.replace(prompt, '').strip()
    try:
        label_ = text.split("Explanation:")[0].lstrip()
        if "Entails" in label_:
            predictedlabel = "Entailment"
        elif "Contradicts" in label_:
            predictedlabel = "Contradiction"
    except:
        predictedlabel = "Contradiction"
    finally:
        data[i]["predicted_label"] = predictedlabel

    try:
        explanation_ = text.split("Explanation:")[1].lstrip()
        try:
            explanation_ = explanation_.split("premise:")[0].lstrip()
        except:
            explanation_ = explanation_
    except:
        explanation_ = "Not given"
    finally:
        data[i]["model_explanation"] = explanation_

with open("prediction2_t0_k0.json","w") as f:
    f.write(json.dumps(data,indent=4))

del data

with open("FLUTE_data/FLUTE_val_2.json") as f:
    data = json.load(f)

for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\nThe output must strictly follow the following format: Label: 'Entails' or 'Contradicts', Explanation: 'text'"
    request = "\npremise: "+premise+"\nhypothesis: "+hypothesis
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=True, temperature=0.3, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=256)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    text = text.replace(prompt, '').strip()
    try:
        label_ = text.split("Explanation:")[0].lstrip()
        if "Entails" in label_:
            predictedlabel = "Entailment"
        elif "Contradicts" in label_:
            predictedlabel = "Contradiction"
    except:
        predictedlabel = "Contradiction"
    finally:
        data[i]["predicted_label"] = predictedlabel

    try:
        explanation_ = text.split("Explanation:")[1].lstrip()
        try:
            explanation_ = explanation_.split("premise:")[0].lstrip()
        except:
            explanation_ = explanation_
    except:
        explanation_ = "Not given"
    finally:
        data[i]["model_explanation"] = explanation_

with open("prediction2_t03_k0.json","w") as f:
    f.write(json.dumps(data,indent=4))
