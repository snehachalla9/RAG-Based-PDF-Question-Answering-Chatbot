import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Q&A Chat", page_icon="📄")
st.title("📘 Ask Your PDF - Chat")


if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    os.makedirs("tempDir", exist_ok=True)
    pdf_path = f"tempDir/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded!")

    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(pages)


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    vectorstore = Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name="pdf_chunks",
        drop_old=True
    )

    retriever = vectorstore.as_retriever()

    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    
    def handle_query():
        query = st.session_state.query_input
        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain.invoke(query)["result"]

            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.session_state.query_input = ""

    
    st.text_input("Ask something:", key="query_input", on_change=handle_query)

    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**🧑 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Bot:** {msg['content']}")
