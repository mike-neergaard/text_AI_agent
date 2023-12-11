from dotenv import load_dotenv
import vec_utils
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

def retrieval_qa(llm, retriever, question: str, answer_length: 1000, verbose: bool = False):
    """
    This chain is used to answer the intermediate questions.
    """
    prompt_answer_length = f" Answer as succinctly as possible in less than {answer_length} words.\n"

    prompt_template = \
        "You are provided with a question and some helpful context to answer the question \n" \
        " Question: {question}\n" \
        " Context: {context}\n" \
        "Your task is to answer the question based in the information given in the context" \
        " Answer the question entirely based on the context and no other previous knowledge." \
        " If the context provided is empty or irrelevant, just return 'Context not sufficient'"\
        + prompt_answer_length

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
        verbose = verbose,
    )

    result = qa_chain({"query": question})
    return result['result'], result['source_documents']

load_dotenv()
TE = vec_utils.TextEmbed()
TE.load_vs("faiss_index")

llm=HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":1, "max_length":1000000})

question = "Who was Moses?"

#qa_chain = RetrievalQA.from_chain_type(\
#        llm=llm, \
#        retriever=TE.vs.as_retriever(search_kwargs={"k": 3})\
#    )
answer = retrieval_qa(llm, \
        TE.vs.as_retriever(search_kwargs={"k": 3}),\
        question, 1000)

#result = qa_chain({"query":question})
print(answer)


