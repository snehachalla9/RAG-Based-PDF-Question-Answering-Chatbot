# 📄 RAG-Based PDF Question Answering Chatbot

An AI-powered chatbot that allows users to upload PDFs and ask questions using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Features
- 📄 Upload and process PDF documents
- 🔍 Semantic search using vector embeddings
- 🤖 Context-aware answers using Google Gemini
- 💬 Conversational interface with Streamlit
- ⚡ Fast retrieval using Milvus vector database

---

## 🧠 Tech Stack
- Python
- Streamlit
- LangChain
- Milvus (Vector Database)
- Google Gemini (gemini-1.5-flash)
- HuggingFace Sentence Transformers

---

## ⚙️ How It Works
1. Upload a PDF
2. Text is split into chunks
3. Convert chunks into embeddings
4. Store embeddings in Milvus
5. Retrieve relevant chunks based on query
6. Gemini generates context-based answers

---

## 🛠️ Installation

```bash
git clone https://github.com/snehachalla9/RAG-Based-PDF-Question-Answering-Chatbot.git
cd RAG-Based-PDF-Question-Answering-Chatbot
pip install -r requirements.txt
