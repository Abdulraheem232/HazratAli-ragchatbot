from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Extracting data 
file_path = "data/"
def extract_data(file_path):
    data_loader = DirectoryLoader(
        file_path,
        glob = "*.pdf",
        loader_cls=PyPDFLoader
    )
    
    return data_loader.load()

# Creating chunks out of data
def create_chunks(data):
    chunks_data = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    chunks_data = chunks_data.split_documents(data)
    return chunks_data

# Create emebedding model
def create_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Create vector store
document = create_chunks(extract_data(file_path))
embedding_model = create_embedding_model()
vector_store_path = "vector_store/faiss_index"
vectorstore = FAISS.from_documents(document,embedding_model)
vectorstore.save_local(vector_store_path)