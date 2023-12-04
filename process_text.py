import os
from transformers import AutoTokenizer
import spacy
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

class chunk_utils:
    """A class of text chunk-managing utilities"""
    previous_chunk = ""
    previous_chunk_tokens = 0
    current_chunk = "" 
    current_chunk_tokens = 0
    line_number = 0
    
    # What percentage of max_tokens do we look for in each text chunk?
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

    def absolute_chunk_reset(self):
        self.previous_chunk = ""
        self.previous_chunk_tokens = 0
        self.current_chunk = ""
        self.current_chunk_tokens = 0

    def add_line_to_chunk(self,line: str):
        self.current_chunk += line
        self.current_chunk_tokens = self.count_tokens(self.current_chunk)

    def start_new_chunk(self, line: str):
        self.current_chunk = line
        self.current_chunk_tokens = self.count_tokens(self.current_chunk)

    def dump_chunk(self, chunk_list): 
        # If there is a previous chunk, write it out
        if self.previous_chunk != "":
            chunk_list.append({"text":self.previous_chunk, 
                "len": self.previous_chunk_tokens, 
                "line number": self.line_number}) 
            self.line_number += 1 

        # Move current_chunk to previous_chunk
        self.previous_chunk = self.current_chunk
        self.current_chunk = ""

        self.previous_chunk_tokens = self.current_chunk_tokens
        self.current_chunk_tokens = 0

    def absolute_dump_chunk(self, chunk_list):
        """ A routine to dump both the previous and current chunk """

        # Concatenate the previous chunk and the current chunk
        big_chunk = self.previous_chunk + self.current_chunk
        big_chunk_tokens = self.count_tokens(big_chunk)

        # Before we get involved in anything else, remember to clear chunks
        self.absolute_chunk_reset()

        # Perhaps the concatenated chunk is short enough
        if big_chunk_tokens < self.max_tokens:
            if big_chunk != "":
                # Apparently there is at least some text
                chunk_list.append({"text":big_chunk,
                    "len": big_chunk_tokens,
                    "line number": self.line_number}) 
                self.line_number += 1
            return

        # The concatenated chunk is too long. We will split it
        chunks=self.chunk_line(big_chunk, big_chunk_tokens)
        # Write the chunks out
        for c in chunks:
            chunk_list.append({"text":c[0], "len": c[1], 
                "line number":self.line_number})
            self.line_number += 1


        

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
    
    def record_chunk(self, line: str, chunk_list: list):
        """ We have to write out a chunk"""
    
        line_tokens = self.count_tokens(line)
        # If the current line is too long, then things get complicated
        # We want to chunk the current line without prepending or appending text
        # from other lines
        if line_tokens > self.max_tokens:
            # First, get rid of the existing chunks 
            self.absolute_dump_chunk(chunk_list)

            # This line is itself too long, so we will have to chunk it 
            chunks=self.chunk_line(line, line_tokens)
            # Write the chunks out directly
            for c in chunks:
                chunk_list.append({"text":c[0], "len": c[1], 
                    "line number":self.line_number})
                self.line_number += 1
            return

        # The current line is not too long
        # If there is a current chunk, write it out.
        if self.current_chunk_tokens: 
            self.dump_chunk(chunk_list)
    
        # If the line is empty, we are done here
        if line=="": return
    
        # The line is not empty, so start a new chunk with this line
        self.start_new_chunk(line)



def process_text():
    ch = chunk_utils()
    file_list = [f for f in os.listdir("text") \
           if os.path.isfile(os.path.join("text", f))]
    file_list.sort()

    chunk_list = []
    for file in file_list:
        colon_split = file.split(":")
        source_number = int(colon_split[0])
        source_name = colon_split[1].split(".")[0].strip()
        print("Processing", source_name, "...", end="", flush=True)
    
        with open(os.path.join("text", file), "r") as book:
            # We might have to concatenate lines into chunks
            # This line is probably superfluous.  Still, good to be sure
            ch.absolute_chunk_reset()
    
            for line in book:
                # We tokenize the concatenated string, which
                # may tokenize differently than two individual strings
                num_tokens = ch.count_tokens(ch.current_chunk + line)
                if num_tokens > ch.min_tokens: 
                    # With this line, we have exceeded the threshold, so we must
                    # record a chunk
                    ch.record_chunk(line, chunk_list)
                else:
                    # Even with this line added, we have not exceeded the threshold
                    ch.add_line_to_chunk(line)
    
            # Whatever is left needs to get written out
            ch.absolute_dump_chunk(chunk_list)
        print("Done", flush=True)

    with open(os.path.join("data", "chunks.json"), "w") as outfile:
        json.dump(chunk_list, outfile, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_text()
