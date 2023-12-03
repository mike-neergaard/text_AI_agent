import os
from transformers import AutoTokenizer
import json


""" Check that text is correctly processed from data """
with open("model_config.json", "r") as infile:
    models = json.load(infile)

tokenizer = AutoTokenizer.from_pretrained(models["text_tokenizer"])
max_tokens = tokenizer.model_max_length

def count_tokens(s: str) -> int:
    """Returns the number of tokens in a text string."""
    tokens = tokenizer(s)
    return len(tokens["input_ids"])

with open(os.path.join("data", "lines.json"), "r") as infile:
    lines = json.load(infile)

checks_passed = True
# Double check that the tokenization matches what we expect
for l in lines:
    num_tokens = count_tokens(l["text"])
    if num_tokens > max_tokens: checks_passed = False
    if num_tokens != l["len"]:
        checks_passed = False
        print("Calculated length not equal to recorded length:\n",\
                num_tokens,"!=",l["len"],"\n",l["text"])

if checks_passed: print("All checks passed")

