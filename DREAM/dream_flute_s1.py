# Script for testing "allenai/System1_FigLang2022"
# Apply the dream_flute model (system1) to FLUTE test set and verify the results

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

model = AutoModelForSeq2SeqLM.from_pretrained("allenai/System1_FigLang2022", torch_dtype=torch.float16, device_map=device)
tokenizer = AutoTokenizer.from_pretrained("t5-3b", torch_dtype=torch.float16, device_map=device)

with open("FLUTE_data/FLUTE_test.json") as f:
    data = json.load(f)

for i in range(len(data)):
    premise = data[i]["premise"]
    hypothesis = data[i]["hypothesis"]
    input_string = "Premise: "+premise+" Hypothesis: "+hypothesis+". Is there a contradiction or entailment between the premise and hypothesis?"
    input_ids = tokenizer.encode(input_string, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=200)
    text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    data[i]["output_dream"] = text
    #["Answer : Contradiction. Explanation : Most people would not be happy to see someone else's new car that they cannot afford because it is way out of their budget"]
    try:
        label = text.split(". Explanation")[0].lstrip()
        label = label.split("Answer :")[1].lstrip()
        data[i]["predicted_label"] = label
        explanation = text.split("Explanation :")[1].lstrip()
        data[i]["model_explanation"] = explanation
    except:
        data[i]["predicted_label"] = "to do"
        data[i]["model_explanation"] = "to do"    

with open("dream_flute_system1.json","w") as f:
    f.write(json.dumps(data,indent=4))
