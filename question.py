from dotenv import load_dotenv
import vec_utils
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

load_dotenv()
TE = vec_utils.TextEmbed()
TE.load_vs("faiss_index")

#tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
#model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":1, "max_length":1000000})

question = "Why did Moses not enter the promised land?"
#q_vec = TE.question_to_vec(question)
#print(q_vec.shape)

qa_chain = RetrievalQA.from_chain_type(\
        llm=llm, \
        retriever=TE.vs.as_retriever(search_kwargs={"k": 3})\
    )
#D, I = TE.vs.search(q_vec, 10)
#TE.load_chunks()

result = qa_chain({"query":question})
print(result)
