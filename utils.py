from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embeddings(texts):
    return model.encode(texts)
