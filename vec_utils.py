from InstructorEmbedding import INSTRUCTOR
import faiss
import os
import numpy as np
import json
import time

class TextEmbed:
    embed_model = INSTRUCTOR('hkunlp/instructor-large')
    question_instruction = 'Represent the question for retrieving supporting documents: '
    embed_instruction = 'Represent the document for retrieval: '
    
    def question_to_vec(self, question):
        return self.embed_model.encode([[self.question_instruction, 
            question]])

    def text_to_vecs(self, filename, vec_filename=""):
        begin_time = time.time()
        with open(os.path.join("data", filename), "r")  as infile:
            chunks = json.load(infile)

        customized_embeddings = []
        i = 0
        total_chunks = len(chunks)
        for c in chunks:
            if i % 10 == 0: 
                print("\rConverted "+str(i),"text chunks of",total_chunks,\
                        "to vectors", flush=True, end="")
            text_to_embed = [[self.embed_instruction, c["text"]]]
            embedding = self.embed_model.encode(text_to_embed)[0]
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
    
    def vecs_to_store(self, array_embeddings, file=False, filename=""):
        self.vs = faiss.IndexFlatL2(array_embeddings.shape[1])
        self.vs.add(array_embeddings)

        if file:
            faiss.write_index(self.vs, os.path.join("data", filename))

    def load_chunks(self):
        with open(os.path.join("data", "chunks.json"), "r") as infile:
            self.chunks = json.load(infile)

    def load_vs(self, filename):
        self.vs = faiss.read_index(os.path.join("data", filename))


def run_pipeline():
    vec_utils = TextEmbed()
    #array_embeddings = vec_utils.text_to_vecs("chunks.json", "raw_vecs.txt")
    #array_embeddings = np.loadtxt(os.path.join("data", "raw_vecs.txt"))
    #vec_utils.vecs_to_store(array_embeddings, file=True, filename="faiss_index")
    vec_utils.load_vs("faiss_index")
    q_vec = vec_utils.question_to_vec("What happened to Sodom?")
    print(q_vec.shape)
    D, I = vec_utils.vs.search(q_vec, 10)
    vec_utils.load_chunks()
    for i in I[0]:
        print(vec_utils.chunks[i])


if __name__ == "__main__":
    run_pipeline()
