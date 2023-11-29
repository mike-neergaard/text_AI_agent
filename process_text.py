import os
from transformers import AutoTokenizer
import spacy
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


with open("model_config.json", "r") as infile:
    models = json.load(infile)

tokenizer = AutoTokenizer.from_pretrained(models["text_tokenizer"])

# spacy seems to give better results in this context than langchain 
s_nlp = spacy.load('en_core_web_trf')
max_tokens = tokenizer.model_max_length

# Not as good as spacy for this purpose
#line_splitter = RecursiveCharacterTextSplitter( 
#        chunk_size = 100, #overwrite
#        chunk_overlap = 20,
#        length_function = count_tokens)

file_list = [f for f in os.listdir("text") \
        if os.path.isfile(os.path.join("text", f))]
file_list.sort()

def count_tokens(s: str) -> int:
    """Returns the number of tokens in a text string."""
    tokens = tokenizer(s)
    return len(tokens["input_ids"])

def chunk_line(line: str, num_tokens: int, 
        margin: int = int(max_tokens * 0.10)) -> list: 

    # Let spacy turn this line into sentences
    s_line = s_nlp(line)
    sentences = [str(i) for i in s_line.sents]

    num_chunks = int(num_tokens / max_tokens) + 1
    # Give a margin for the maximum chunk size
    # But not larger than max_tokens
    chunk_size = min(int(num_tokens / num_chunks) + margin, max_tokens)

    chunks = []
    current_chunk = ""
    for i in range(len(sentences)): 
        # Extend the current chunk by one sentence
        new_chunk = current_chunk + sentences[i]
        if count_tokens(new_chunk) > chunk_size:
            # Time to start a new chunk
            chunks.append([current_chunk, count_tokens(current_chunk)])
            current_chunk = sentences[i]
        else:
            # Update current chunk and continue
            current_chunk = new_chunk

    chunks.append([current_chunk, count_tokens(current_chunk)])

    return chunks
    

line_number = 0
lines = []
for file in file_list:
    colon_split = file.split(":")
    source_number = int(colon_split[0])
    source_name = colon_split[1].split(".")[0].strip()
    print("Processing", source_name, "...", end="", flush=True)

    with open(os.path.join("text", file), "r") as book:
        for line in book:
            num_tokens = count_tokens(line)
            if num_tokens > max_tokens:
                # This line is too long, so we will have to chunk it 
                chunks=chunk_line(line, num_tokens)
                for c in chunks:
                    lines.append({"text":c[0], "len": c[1], "line number":line_number})
                    line_number += 1
            else:
                lines.append({"text":line, "len": num_tokens, "line number":line_number})
                line_number += 1
    print("Done", flush=True)

with open(os.path.join("data", "lines.json"), "w") as outfile:
    json.dump(lines, outfile, ensure_ascii=False, indent=2)

