# Script for applying DREAM at FLUTE Dataset
# Take the validation set and the test set and insert for each example the Scene Elaboration given by Dream
# The "contest" is given by these four elements:
#  1. M : motivation of character(s) before S.
#  2. E: emotion of character(s) after S.
#  3. ROT : general Rule of Thumb (ROT) about whether action described in S is socially acceptable or not (also known as social norm).
#  4. Con: likely consequence of action in S

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

dream_dim = ["motivation", "emotion", "rot", "consequence"]

# Validation Set
with open("home/solivero/FLUTE_data/FLUTE_val.json") as f:
    data = json.load(f)

for k in dream_dim:
  for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    situation = "Premise: '"+premise+"'. Hypothesis: '"+hypothesis+"'."
    input_string = "$answer$ ; $question$ = [SITUATION] "+situation+" [QUERY] "+k
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    output = model.generate(input_ids, max_length=200)
    text = tokenizer.batch_decode(output, skip_special_tokens=True)
    #["$answer$ = It's wrong to damage other people's property."]  
    text = text.split("$answer$ =")[1].lstrip()
    data[i][k] = text

with open("home/solivero/dream_val.json","w") as f:
    f.write(json.dumps(data,indent=4))

del data

# Test Set
with open("home/solivero/FLUTE_data/FLUTE_test.json") as f:
    data = json.load(f)

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
