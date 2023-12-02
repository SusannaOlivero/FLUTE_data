import torch
import transformers
import numpy as np
import random
import json

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device('cuda')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("allenai/DREAM", torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained("t5-11b", torch_dtype=torch.float16, device_map=device)

# Validation Set
with open("home/solivero/FLUTE_data/FLUTE_val.json") as f:
    data = json.load(f)
    
dream_dim = ["motivation", "emotion", "rot", "consequence"]

for k in dream_dim:
  for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    situation = "Premise: '"+premise+"'. Hypothesis: '"+hypothesis+"'."
    input_string = "$answer$ ; $question$ = [SITUATION] "+situation+" [QUERY] "+k
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    output = model.generate(input_ids, max_length=200)
    text = tokenizer.batch_decode(output, skip_special_tokens=True)
    text = text.split("$answer$ =")[1].lstrip()
    data[i][k] = text

with open("home/solivero/dream_val.json","w") as f:
    f.write(json.dumps(data,indent=4))

del data

# Test Set
with open("home/solivero/FLUTE_data/FLUTE_test.json") as f:
    data = json.load(f)
    
dream_dim = ["motivation", "emotion", "rot", "consequence"]

for k in dream_dim:
  for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    situation = "Premise: '"+premise+"'. Hypothesis: '"+hypothesis+"'."
    input_string = "$answer$ ; $question$ = [SITUATION] "+situation+" [QUERY] "+k
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    output = model.generate(input_ids, max_length=200)
    text = tokenizer.batch_decode(output, skip_special_tokens=True)
    text = text.split("$answer$ =")[1].lstrip()
    data[i][k] = text

with open("home/solivero/dream_test.json","w") as f:
    f.write(json.dumps(data,indent=4))
