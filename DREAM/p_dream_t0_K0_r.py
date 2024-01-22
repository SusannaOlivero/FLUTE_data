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

## PROMPT style 2
with open("FLUTE_data/DREAM/DreamData/dream_test_SE.json") as f:
    data_val = json.load(f)

# 1.1) Emotion
data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    request = "\npremise: "+data[i]["premise"]+" emotion: "+data[i]["premise-emotion"]
    request += "\nhypothesis: "+data[i]["hypothesis"]+" emotion: "+data[i]["hypothesis-emotion"]
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=False, temperature=0, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=300)
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

with open("p_dream_t0_k0_emotion.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data

# 1.2) Motivation
data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    request = "\npremise: "+data[i]["premise"]+" motivation: "+data[i]["premise-motivation"]
    request += "\nhypothesis: "+data[i]["hypothesis"]+" motivation: "+data[i]["hypothesis-motivation"]
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=False, temperature=0, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=300)
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

with open("p_dream_t0_k0_motivation.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data

# 1.3) Social Norm 
data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    request = "\npremise: "+data[i]["premise"]+" social-norm: "+data[i]["premise-rot"]
    request += "\nhypothesis: "+data[i]["hypothesis"]+" social-norm: "+data[i]["hypothesis-rot"]
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=False, temperature=0, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=300)
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

with open("p_dream_t0_k0_rot.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data

# 1.4) Consequence
data = data_val
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    request = "\npremise: "+data[i]["premise"]+" consequence: "+data[i]["premise-consequence"]
    request += "\nhypothesis: "+data[i]["hypothesis"]+" consequence: "+data[i]["hypothesis-consequence"]
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=False, temperature=0, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=300)
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

with open("p_dream_t0_k0_consequence.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data

# 2) Scene Elaboration completed
data = data_val
pm = "premise-motivation"
pe = "premise-emotion"
pr = "premise-rot"
pc = "premise-consequence"
hm = "hypothesis-motivation"
he = "hypothesis-emotion"
hr = "hypothesis-rot"
hc = "hypothesis-consequence"
for i in range(len(data)):
    prompt = "Find if the 'premise' entails or contradicts the 'hypothesis'.\n"
    prompt += "Let's think step by step.\n"
    request = f"\npremise: {data[i]['premise']} [motivation] {data[i][pm]} [emotion] {data[i][pe]} [social norm] {data[i][pr]} {data[i][pc]}\n"
    request += f"hypothesis: {data[i]['hypothesis']} [motivation] {data[i][hm]} [emotion] {data[i][he]} [social norm] {data[i][hr]} {data[i][hc]}\n"
    prompt += request
    inputs = tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to(device)
    generate_ids = model.generate(**inputs, do_sample=False, temperature=0, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=350)
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

with open("p_dream_t0_k0_SE.json","w") as f:
    f.write(json.dumps(data,indent=4))
del data
