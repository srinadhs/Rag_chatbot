import io
import os
os.environ["STREAMLIT_WATCH"] = "false"
import streamlit as st
import numpy as np
import faiss
from PyPDF2 import PdfReader
import google.generativeai as genai
os.environ["STREAMLIT_WATCH"] = "false"


# OCR support
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# -----------------------
# Config
# -----------------------
API_KEY = "you google gemini 1.5 flash api key here"

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    API_KEY = os.environ["GOOGLE_API_KEY"]
except (KeyError, ValueError):
    genai.configure(api_key=API_KEY)
    if not API_KEY:
        st.error("API Key not found. Set GOOGLE_API_KEY environment variable or provide it in code.")
        st.stop()
    st.info("Using placeholder API key. For production, set GOOGLE_API_KEY environment variable.")

EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"

# -----------------------
# Helper Functions
# -----------------------
def extract_text_from_pdf_bytes(file_bytes: bytes):
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception:
        pass

    text = text.strip()
    if text:
        return text

    if OCR_AVAILABLE:
        images = convert_from_bytes(file_bytes)
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img) + "\n"
        return ocr_text.strip()
    
    return ""

def read_txt_bytes(file_bytes: bytes):
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def chunk_text(text: str, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

@st.cache_data
def embed_texts(texts):
    embeddings = []
    for i, t in enumerate(texts):
        try:
            resp = genai.embed_content(model=EMBED_MODEL, content=t)
            vec = resp.get("embedding") or resp.get("data")[0].get("embedding")
            embeddings.append(np.array(vec, dtype="float32"))
        except Exception as e:
            st.error(f"Embedding failed for chunk {i}: {e}")
            continue
    if not embeddings:
        return None
    return np.vstack(embeddings)

@st.cache_data
def build_faiss_index(embeddings: np.ndarray):
    if embeddings is None or embeddings.shape[0] == 0:
        return None
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype("float32"))
    return index

def retrieve_top_k(index, chunks, query, top_k=3):
    try:
        qvec_resp = genai.embed_content(model=EMBED_MODEL, content=query)
        qvec = qvec_resp.get("embedding") or qvec_resp.get("data")[0].get("embedding")
        q = np.array(qvec, dtype="float32").reshape(1, -1)
        D, I = index.search(q, top_k)
        retrieved = [(chunks[int(idx)], float(D[0][i])) for i, idx in enumerate(I[0]) if idx >= 0]
        return retrieved
    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        return []

def generate_with_context(question, contexts=None, chat_history=None):
    context_text = "\n\n".join(contexts) if contexts else ""
    
    history_text = ""
    if chat_history:
        for msg in chat_history:
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"
    
    prompt = f"""You are a helpful and friendly AI assistant.
{f'Use the context below to answer the question:\n{context_text}' if context_text else ''}
Chat History:
{history_text}
User: {question}
Assistant:"""
    
    try:
        model = genai.GenerativeModel(CHAT_MODEL)
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return "Sorry, I am unable to generate a response at this time."

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="RAG Chat + Casual AI", page_icon="📄", layout="wide")
st.title("📄 RAG Chatbot")

# About / AI Introduction
st.markdown(
    """
    **About:**  
    This AI chatbot was created by **Sreenadh S** using **Google Gemini AI**.  
    It can answer questions from uploaded PDF/TXT documents (RAG mode) or chat casually if no documents are uploaded.
    """
)

# Session state
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_names" not in st.session_state:
    st.session_state.file_names = []

# Initial AI self-introduction
if not st.session_state.messages:
    intro_message = "Hello! I’m an AI assistant created by **Sreenadh S** using **Google Gemini AI**. I can help answer questions from uploaded documents (RAG mode) or chat casually with you."
    st.session_state.messages.append({"role": "assistant", "content": intro_message})

# Sidebar
with st.sidebar:
    st.header("Document & Settings")
    uploaded_files = st.file_uploader("Upload PDF/TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Build Index"):
        if not uploaded_files:
            st.warning("Please upload files first.")
        else:
            all_text = []
            file_names = []
            with st.spinner("Processing files..."):
                for f in uploaded_files:
                    name = f.name
                    file_names.append(name)
                    file_bytes = f.read()
                    if name.lower().endswith(".pdf"):
                        text = extract_text_from_pdf_bytes(file_bytes)
                    else:
                        text = read_txt_bytes(file_bytes)
                    if text.strip():
                        all_text.append(text)
            
            if not all_text:
                st.error("No text extracted from uploaded files.")
            else:
                with st.spinner("Chunking text..."):
                    chunks = []
                    for doc_text in all_text:
                        chunks.extend(chunk_text(doc_text))
                    st.session_state.chunks = chunks
                
                if not st.session_state.chunks:
                    st.error("Text chunking resulted in no chunks. Check your document content.")
                else:
                    with st.spinner(f"Creating embeddings for {len(st.session_state.chunks)} chunks..."):
                        embeddings = embed_texts(st.session_state.chunks)
                    
                    with st.spinner("Building FAISS index..."):
                        faiss_index = build_faiss_index(embeddings)
                    
                    if faiss_index is not None:
                        st.session_state.faiss_index = faiss_index
                        st.session_state.file_names = file_names
                        st.success(f"FAISS index built with {len(st.session_state.chunks)} chunks.")
                    else:
                        st.error("Failed to build FAISS index.")
    
    # Clear Chat and Index
    if st.button("Clear Chat and Index"):
        st.session_state.messages = []
        st.session_state.chunks = []
        st.session_state.faiss_index = None
        st.session_state.file_names = []
        st.success("Chat and index cleared. You can rebuild the index now.")
    
    st.markdown("---")
    st.subheader("RAG Settings")
    top_k = st.slider("Number of chunks to retrieve (Top-k)", min_value=1, max_value=10, value=3)
    
    if st.session_state.file_names:
        st.markdown("**Indexed Files:**")
        for file_name in st.session_state.file_names:
            st.write(f"- {file_name}")
    else:
        st.info("Upload files and build index to enable RAG.")

# -----------------------
# Display chat history and user input
# -----------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question...", key="main_chat_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            if st.session_state.faiss_index and st.session_state.chunks:
                retrieved = retrieve_top_k(
                    st.session_state.faiss_index, 
                    st.session_state.chunks, 
                    prompt, 
                    top_k=top_k
                )
                
                if retrieved:
                    chunks_only = [c[0] for c in retrieved]
                    st.markdown("**Using RAG with the following context:**")
                    for i, (chunk, dist) in enumerate(retrieved):
                        st.caption(f"Chunk {i+1} — distance: {dist:.4f}")
                        st.code(chunk[:300] + "...", language="markdown")
                    
                    answer = generate_with_context(prompt, contexts=chunks_only, chat_history=st.session_state.messages)
                else:
                    st.warning("No relevant chunks found. Generating a general response.")
                    answer = generate_with_context(prompt, chat_history=st.session_state.messages)
            else:
                answer = generate_with_context(prompt, chat_history=st.session_state.messages)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
