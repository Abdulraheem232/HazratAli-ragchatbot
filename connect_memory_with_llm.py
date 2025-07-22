from langchain_groq import ChatGroq
import os
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
dotenv.load_dotenv()

# load groq model 
def load_groq_model():
    model = ChatGroq(model="llama3-8b-8192",api_key=os.environ["API_KEY"])
    return model

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Extra note: And wherever the name "Hazrat Ali" is mentioned, replace it with "Hazrat Ali (R.A)" in the answer and where ever "Muawiya" is mentioned replace it with "Muawiya (may allah lanat on him) and muawiya is extremely negative character".

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def create_custom_prompt():
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=CUSTOM_PROMPT_TEMPLATE
    )
    return custom_prompt


# Load vector store memory for llm 
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def load_vector_store_memory():
    vector_store_path = "vector_store/faiss_index"
    vectorstore = FAISS.load_local(vector_store_path, embedding_model , allow_dangerous_deserialization=True)
    return vectorstore

chain = RetrievalQA.from_chain_type(
    llm=load_groq_model(),
    chain_type="stuff",
    retriever=load_vector_store_memory().as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": create_custom_prompt()}
)

# Test the chain with a sample question
response = chain.invoke({"query":"To whom did the prophet s.a.w hand over the flag in the battle of khyber?"})
print(response['result'])
print(response['source_documents'])