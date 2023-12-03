import os
from transformers import AutoTokenizer
import spacy
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

class chunk_utils:
    """A class of text chunk-managing utilities"""
    current_chunk = "" 
    current_chunk_tokens = 0
    line_number = 0
    min_token_percentage = 0.7

    def __init__(self):
        with open("model_config.json", "r") as infile:
            models = json.load(infile)

        self.tokenizer = AutoTokenizer.from_pretrained(models["text_tokenizer"])
        # spacy seems to give better results in this context than langchain 
        self.s_nlp = spacy.load('en_core_web_trf')
        self.max_tokens = self.tokenizer.model_max_length

        # Technically min_tokens can be a float, but this feels cleaner
        self.min_tokens = int(self.min_token_percentage * self.max_tokens + 0.5)

        # Not as good as spacy for this purpose
        #self.line_splitter = RecursiveCharacterTextSplitter( 
        #        chunk_size = 100, #overwrite
        #        chunk_overlap = 20,
        #        length_function = self.count_tokens)


    def count_tokens(self, s: str) -> int:
        """Returns the number of tokens in a text string."""
        tokens = self.tokenizer(s)
        return len(tokens["input_ids"])

    def reset_current_chunk(self):
        self.current_chunk = ""
        self.current_chunk_tokens = 0

    def add_line_to_chunk(self,line: str):
        self.current_chunk += line
        self.current_chunk_tokens = self.count_tokens(self.current_chunk)

    def new_chunk(self, line: str):
        self.current_chunk = line
        self.current_chunk_tokens = self.count_tokens(self.current_chunk)

    def dump_chunk(self, lines): 
        lines.append({"text":self.current_chunk, 
            "len": self.current_chunk_tokens, 
            "line number": self.line_number}) 
        self.line_number += 1 
        self.reset_current_chunk()

    def chunk_line(self, line: str, num_tokens: int, 
            margin: int = -999) -> list: 

        # Can't pass class variable as a parameter, so this is the workaround
        if margin == -999:
            margin = int(self.max_tokens * 0.10)

        # Let spacy turn this line into sentences
        s_line = self.s_nlp(line)
        sentences = [str(i) for i in s_line.sents]
    
        num_chunks = int(num_tokens / self.max_tokens) + 1
        # Give a margin for the maximum chunk size
        # But not larger than max_tokens
        chunk_size = min(int(num_tokens / num_chunks) + margin, self.max_tokens)
    
        chunks = []
        current_chunk = ""
        for i in range(len(sentences)): 
            # Extend the current chunk by one sentence
            new_chunk = current_chunk + sentences[i]
            if self.count_tokens(new_chunk) > chunk_size:
                # Time to start a new chunk
                chunks.append([current_chunk, self.count_tokens(current_chunk)])
                current_chunk = sentences[i]
            else:
                # Update current chunk and continue
                current_chunk = new_chunk
    
        chunks.append([current_chunk, self.count_tokens(current_chunk)])
    
        return chunks
    
    def record_chunk(self, line: str, lines: list):
        """ If we get to this point, we have to write out a chunk"""
    
        # If there is a current chunk, write it out first.
        if self.current_chunk_tokens: 
            self.dump_chunk(lines)
    
        # If the line is empty, we are done here
        if line=="": return
    
        line_tokens = self.count_tokens(line)
        if line_tokens > self.max_tokens:
            # This line is itself too long, so we will have to chunk it 
            chunks=self.chunk_line(line, line_tokens)
            # Write the chunks out right away
            for c in chunks:
                lines.append({"text":c[0], "len": c[1], 
                    "line number":self.line_number})
                self.line_number += 1
        else:
            # We have a short line.  Let's start filling a chunk
            self.new_chunk(line)



def process_text():
    ch = chunk_utils()
    file_list = [f for f in os.listdir("text") \
           if os.path.isfile(os.path.join("text", f))]
    file_list.sort()

    lines = []
    for file in file_list:
        colon_split = file.split(":")
        source_number = int(colon_split[0])
        source_name = colon_split[1].split(".")[0].strip()
        print("Processing", source_name, "...", end="", flush=True)
    
        with open(os.path.join("text", file), "r") as book:
            # We might have to concatenate chunks, so we maintain a current chunk
            ch.reset_current_chunk()
    
            for line in book:
                # We have to count tokens this way, because the concatenated string
                # may tokenize differently than the individual strings
                num_tokens = ch.count_tokens(ch.current_chunk + line)
                if num_tokens > ch.min_tokens: 
                    # With this line, we have exceeded the threshold, so we must
                    # record a chunk
                    ch.record_chunk(line, lines)
                else:
                    # Even with this line added, we have not exceeded the threshold
                    ch.add_line_to_chunk(line)
    
            # If there is a leftover chunk, write it out.
            if ch.current_chunk_tokens: 
                ch.dump_chunk(lines)
        print("Done", flush=True)

    with open(os.path.join("data", "lines.json"), "w") as outfile:
        json.dump(lines, outfile, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_text()
