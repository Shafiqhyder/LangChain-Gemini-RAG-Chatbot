# 📚 Free University PDF Chatbot (RAG System)

A 100% free, production-ready Retrieval-Augmented Generation (RAG) application built using **LangChain** and **Streamlit**. This tool allows students and developers to upload academic PDFs (books, research papers, or past papers) and have context-aware, intelligent conversations with the document without incurring any API costs.

---

## 🚀 Features
- **Local Text Embeddings:** Uses HuggingFace's open-source `all-MiniLM-L6-v2` model for generating text embeddings locally.
- **Free LLM Integration:** Powered by Google Gemini (`gemini-1.5-flash`) via the free Google AI Studio tier.
- **Smart Chunking:** Implements `RecursiveCharacterTextSplitter` for optimal context splitting and retrieval.
- **In-Memory Vector Store:** Utilizes `ChromaDB` to store and perform similarity searches on document vectors.
- **Interactive UI:** Built with Streamlit for a clean, user-friendly chat experience.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Frameworks:** LangChain, Streamlit
- **Embeddings:** HuggingFace Transformers (`all-MiniLM-L6-v2`)
- **LLM:** Google Gemini API
- **Vector Database:** ChromaDB

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
