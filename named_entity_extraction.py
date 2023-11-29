from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import os
import json

load_dotenv()

with open("model_config.json", "r") as infile:
    models = json.load(infile)

tokenizer = AutoTokenizer.from_pretrained(models["NER_tokenizer"])
model = AutoModelForTokenClassification.from_pretrained(models["NER_model"])
NER = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

with open(os.path.join("data", "lines.json"), "r") as infile:
    lines = json.load(infile)

entities = {}
num_lines = len(lines)
for line in lines:
    if line["line number"] % 10 == 0:
        print("\r"+str(line["line number"]),"of",num_lines, end="", flush=True)
    line_entities = NER(line["text"])
    for e in line_entities:
        group = e["entity_group"]
        name = e["word"]
        if group not in entities:
            entities[group] = {}
        if name not in entities[group]:
            entities[group][name] = [line["line number"]]
        else:
            entities[group][name].append(line["line number"])

print("\r"+str(num_lines),"of",num_lines, end="")
with open(os.path.join("data", "entities.json"), "w") as outfile:
    json.dump(entities, outfile, ensure_ascii = False, indent=2)

