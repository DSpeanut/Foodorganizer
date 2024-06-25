from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp


model_download_path = "~/models/llama-2-7b-chat.Q2_K.gguf"

def vectorstore_transformation(input_text):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_texts([input_text], embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def prompt_setup():
    template = """Answer the question without any explanation based only on the following context: {context}  Question: {question}\
                Given the input text, detect food ingredients and its associated quantities or measurments.\
                Ingredient may have quantity or measurment choose whichever is available.
                If quantity or measurment is not available use 1 as quantity or measurement.\
                Provide output as following format 'ingredient name : quantity or measurement.'\
                Do not print any output of explanation or note.
                Finally print out each item in each line\
                """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def llama27b_chat(temperature, max_tokens, top_p):
    model_path = model_download_path
    llm = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return llm

def langchain_output(retriever, prompt, llm):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm)
    output = chain.invoke("what are the ingredients and quantities or measurements?")
    return output.strip('Response:').strip('Outcome:')

