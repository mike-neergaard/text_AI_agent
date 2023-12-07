#from InstructorEmbedding import INSTRUCTOR
import faiss
import os
import numpy as np
import json
import time
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings


class TextEmbed:
    def __init__(self):
        with open("model_config.json", "r") as infile:
            model = json.load(infile)
        self.embeddings = HuggingFaceInstructEmbeddings(
            query_instruction=model["embed_instruction"]
        )
        self.q_embeddings = HuggingFaceInstructEmbeddings(
            query_instruction=model["question_instruction"]
        )
        self.chunks = []
    
    def question_to_vec(self, question):
        return self.q_embeddings.embed_query(question)

    def text_to_vecs(self, filename, vec_filename=""):
        begin_time = time.time()
        self.load_chunks()

        customized_embeddings = []
        i = 0
        total_chunks = len(self.chunks)
        for c in self.chunks:
            if i % 10 == 0: 
                print("\rConverted "+str(i),"text chunks of",total_chunks,\
                        "to vectors", flush=True, end="")
            embedding = self.embeddings.embed_query(c)
            customized_embeddings.append(embedding)
            i += 1

        print("\rConverted "+str(i),"text chunks of",total_chunks,\
                "to vectors", flush=True)

        array_embeddings = np.array(customized_embeddings)
        if vec_filename != "": 
            np.savetxt(os.path.join("data", vec_filename), array_embeddings)
        print("Process done.  Text to vectors time:", time.time()-begin_time, \
                "seconds")
        return array_embeddings
    
    def vecs_to_store(self, array_embeddings, file=False, index_name="index"):
        self.load_chunks()
        list_of_vecs = [v.tolist() for v in array_embeddings]
        embed_vals = [[self.chunks[i], array_embeddings[i].tolist()]\
                for i in range(len(self.chunks))]
        #self.vs = faiss.IndexFlatL2(array_embeddings.shape[1])
        #self.vs.add(array_embeddings)

        # Create the vector store with the query instruction
        self.faiss = FAISS.from_embeddings(embed_vals, self.q_embeddings)

        if file:
            self.faiss.save_local("data", index_name=index_name)
            #self.faiss.write_index(self.vs, os.path.join("data", filename))

    def load_chunks(self):
        if len(self.chunks) > 0: return
        with open(os.path.join("data", "chunks.json"), "r") as infile:
            chunk_list = json.load(infile)
        self.chunks = [c["text"] for c in chunk_list]

    def load_vs(self, index_name):
        #self.vs = faiss.read_index(os.path.join("data", filename))
        self.vs = FAISS.load_local("data", self.q_embeddings, \
                index_name=index_name)



def run_pipeline():
    vec_utils = TextEmbed()
    #array_embeddings = vec_utils.text_to_vecs("chunks.json", "raw_vecs.txt")
    array_embeddings = np.loadtxt(os.path.join("data", "raw_vecs.txt"))
    vec_utils.vecs_to_store(array_embeddings, file=True, \
            index_name="faiss_index")



if __name__ == "__main__":
    run_pipeline()
