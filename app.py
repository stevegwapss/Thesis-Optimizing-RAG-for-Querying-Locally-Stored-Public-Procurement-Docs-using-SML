# app.py
from flask import Flask, request, jsonify, send_from_directory
import os
import json
import tempfile
import faiss
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import requests
from pathlib import Path

app = Flask(__name__, static_folder='.', static_url_path='')

# Configuration
UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
VECTOR_DIMENSION = 384  # Dimension for embeddings (depends on model used)
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(DB_FOLDER).mkdir(exist_ok=True)

# Initialize embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Small, fast model

# Initialize FAISS index
faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)  # Renamed from 'index' to 'faiss_index'
document_chunks = []  # Store text chunks corresponding to vectors

# Load existing index if available
def load_index():
    global faiss_index, document_chunks  # Change index to faiss_index
    if os.path.exists(f"{DB_FOLDER}/index.faiss") and os.path.exists(f"{DB_FOLDER}/chunks.json"):
        try:
            faiss_index = faiss.read_index(f"{DB_FOLDER}/index.faiss")  # Change index to faiss_index
            with open(f"{DB_FOLDER}/chunks.json", 'r') as f:
                document_chunks = json.load(f)
            print(f"Loaded existing index with {len(document_chunks)} chunks")
        except Exception as e:
            print(f"Error loading index: {e}")
            # Initialize new index
            faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)  # Change index to faiss_index
            document_chunks = []

# Save index
def save_index():
    faiss.write_index(faiss_index, f"{DB_FOLDER}/index.faiss")  # Change index to faiss_index
    with open(f"{DB_FOLDER}/chunks.json", 'w') as f:
        json.dump(document_chunks, f)
    print(f"Index saved with {len(document_chunks)} chunks")

# Text chunking function
def chunk_text(text, filename="", page_num=0):
    chunks = []
    i = 0
    while i < len(text):
        # Get chunk with overlap
        chunk = text[i:i + CHUNK_SIZE]
        if chunk:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "start_char": i,
                    "end_char": min(i + CHUNK_SIZE, len(text))
                }
            })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# Extract text from PDF
def extract_pdf_text(file_path):
    chunks = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    page_chunks = chunk_text(text, os.path.basename(file_path), i)
                    chunks.extend(page_chunks)
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
    return chunks

# Add a document to the index
def add_document_to_index(file_path):
    chunks = extract_pdf_text(file_path)
    if not chunks:
        return {"success": False, "message": "No text extracted from document"}
    
    # Get embeddings for all chunks
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    
    # Add to FAISS index
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    # Store chunk information
    start_idx = len(document_chunks)
    for i, chunk in enumerate(chunks):
        document_chunks.append(chunk)
    
    save_index()
    return {"success": True, "chunks_added": len(chunks)}

# Query Ollama using retrieval augmentation
def query_ollama(query, top_k=3):
    # Get query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS
    D, I = faiss_index.search(query_embedding, top_k)
    
    if len(I[0]) == 0:
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Get relevant contexts
    contexts = []
    for idx in I[0]:
        if idx < len(document_chunks):
            contexts.append(document_chunks[idx]["text"])
    
    # Build prompt with context
    context_text = "\n\n".join(contexts)
    prompt = f"""
    You are an expert in procurement documents. 
    Use the following information to answer the query.
    
    Context information:
    {context_text}
    
    Query: {query}
    
    Answer based only on the provided context. If the information is not in the context, say that you don't know.
    """
    
    # Query Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_gpu": 0  # Force CPU mode
                }
            }
        )
        
        if response.status_code == 200:
            return {"response": response.json()["response"]}
        else:
            return {"error": f"Ollama error: {response.text}"}
    except Exception as e:
        return {"error": f"Error querying Ollama: {str(e)}"}

# Flask routes
@app.route('/')
def serve_index():  # Renamed from 'index' to 'serve_index'
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    # Save file temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)
    
    # Process file
    result = add_document_to_index(temp_path)
    
    return jsonify(result)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    result = query_ollama(data['query'])
    return jsonify(result)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "documents_count": len(set(chunk["metadata"]["source"] for chunk in document_chunks)),
        "chunks_count": len(document_chunks),
        "ollama_status": "connected" if check_ollama_connection() else "disconnected"
    })

def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

# Initialize index on startup
load_index()

if __name__ == '__main__':
    app.run(debug=True, port=5000)