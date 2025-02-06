import streamlit as st 
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Custom CSS for styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
    }
    .stMarkdown h2 {
        margin-bottom: 5px !important;  /* Reduced bottom margin */
        text-align: center !important;
        color: #4CAF50 !important;
        font-weight: 600 !important;
    }
    .stFileUploader {
        border: 2px dashed #00E676 !important;
        border-radius: 10px;
        padding: 10px;
        background-color: #1E1E1E;
    }
    .stTextInput, .stTextArea, .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #00E676 !important;
        border-radius: 5px;
    }
    .stChatMessage:nth-child(odd) {
        background-color: #2E2E2E !important;
        padding: 12px;
        border-radius: 8px;
    }
    .stChatMessage:nth-child(even) {
        background-color: #1E1E1E !important;
        padding: 12px;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #00E676 !important;
        color: #000 !important;
        border-radius: 5px;
        font-weight: bold;
    }
    h5 {
        text-align: center !important;
        margin-top: 5px !important;    /* Reduced top margin */
        margin-bottom: 15px !important; /* Keep some space below h5 */
    }
    
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <h2>📜Insights AI</h2>
    <h5>AI-Powered Document Analysis</h5>
""", unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

# File Upload Section
st.markdown("""<h3>Upload a Document</h3>""", unsafe_allow_html=True)
uploaded_pdf = st.file_uploader(
    "Drop a PDF file here or click to upload",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant"):
            st.write(ai_response)

st.markdown("""
    <h3>How to Use</h3>
    <ul>
        <li><b>Upload PDF:</b> Click on the file uploader and select a PDF document.</li>
        <li><b>Process Document:</b> Wait for the document to be processed.</li>
        <li><b>Ask Questions:</b> Type your query in the chat input.</li>
        <li><b>Get Insights:</b> Receive AI-generated answers based on the document context.</li>
    </ul>
    <hr>
""", unsafe_allow_html=True)
