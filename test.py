import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Free Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI # Free LLM
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration (UI Setup)
st.set_page_config(page_title="Free University PDF Chatbot", page_icon="📚", layout="centered")
st.title("📚 Free University PDF Chatbot (RAG System)")
st.write("Upload your university book or past paper (PDF) and ask questions for FREE!")

# 2. Free Google Gemini API Key Input via Sidebar
# Tip: You can get a free key from Google AI Studio
api_key = st.sidebar.text_input("Enter your FREE Google Gemini API Key:", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

    # 3. File Uploader Component
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded buffer
        with open("temp_pdf_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("File uploaded successfully! Processing the document...")
        
        @st.cache_resource
        def process_pdf(file_path):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            
            # --- ROADMAP STAGE 5: Free Open Source Embeddings ---
            # Yeh model internet se free download hoga aur computer par chalega
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            vector_store = Chroma.from_documents(final_documents, embeddings)
            return vector_store

        vector_store = process_pdf("temp_pdf_file.pdf")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # --- ROADMAP STAGE 2: Free Google Gemini Model ---
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        system_prompt = (
            "You are an expert academic assistant. Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you don't know. Keep the answer concise.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # Helper function to format documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # 4. Chat Interface (Streamlit UI)
        st.write("---")
        user_question = st.text_input("Ask any question regarding your uploaded document:")

        if user_question:
            with st.spinner("AI is searching for the answer..."):
                retrieved_docs = retriever.invoke(user_question)
                context_text = format_docs(retrieved_docs)
                formatted_prompt = prompt.format_messages(context=context_text, question=user_question)
                
                ai_response = llm.invoke(formatted_prompt)
                
                st.write("### ✨ AI Response:")
                st.write(ai_response.content)

else:
    st.warning("Please enter your Google Gemini API Key in the sidebar. You can get one for FREE from Google AI Studio.")
    st.info("Link to get free key: https://aistudio.google.com/")