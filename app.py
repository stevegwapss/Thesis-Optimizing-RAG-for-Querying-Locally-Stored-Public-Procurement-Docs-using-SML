# app_langchain.py - LangChain refactor of the RAG system
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from pathlib import Path
import glob
import threading
import time
from typing import List, Dict, Tuple, Union
from collections import defaultdict
import traceback
import logging
from datetime import datetime
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
CHROMA_DB_PATH = 'db/chromadb'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Create folders if they don't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)

# Global variables for LangChain components
embeddings = None
vectorstore = None
llm = None
text_splitter = None
qa_chains = {}  # Role-specific QA chains

# Progress tracking (keeping existing functionality)
processing_progress = {
    'total_files': 0,
    'processed_files': 0,
    'current_file': '',
    'status': 'idle', 
    'errors': [],
    'start_time': None,
    'folder_path': ''
}
progress_lock = threading.Lock()

def initialize_langchain_components():
    """Initialize LangChain components (basic only, lazy load others)"""
    global embeddings, vectorstore, llm, text_splitter
    
    try:
        # Initialize text splitter first (no network required)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Ollama LLM (no network required if model exists locally)
        llm = OllamaLLM(
            model="llama3.2:3b",
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=0.1
        )
        
        print("LangChain components initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing LangChain components: {e}")
        return False

def get_or_initialize_embeddings():
    """Lazy initialization of embeddings"""
    global embeddings
    if embeddings is None:
        try:
            print("Initializing HuggingFace embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("Embeddings initialized successfully")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            return None
    return embeddings

def get_or_initialize_vectorstore():
    """Lazy initialization of vectorstore"""
    global vectorstore
    if vectorstore is None:
        embeddings_model = get_or_initialize_embeddings()
        if embeddings_model is None:
            return None
            
        try:
            print("Initializing ChromaDB vectorstore...")
            vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings_model,
                collection_name="procurement_docs"
            )
            
            # Initialize role-specific QA chains now that vectorstore is ready
            initialize_qa_chains()
            
            print("Vectorstore initialized successfully")
        except Exception as e:
            print(f"Error initializing vectorstore: {e}")
            return None
    return vectorstore

def initialize_qa_chains():
    """Initialize role-specific QA chains with different prompts"""
    global qa_chains, vectorstore, llm
    
    # Role-specific prompts
    role_prompts = {
        'general': """You are an AI assistant helping users understand procurement documents. 
        Provide clear, comprehensive answers based on the document content.
        
        Context: {context}
        Question: {question}
        
        Answer:""",
        
        'auditor': """You are an AI assistant specialized for auditors reviewing procurement documents.
        Focus on compliance, legal requirements, proper procedures, budget verification, and risk assessment.
        Highlight any potential issues or areas requiring attention.
        
        Context: {context}
        Question: {question}
        
        Auditor-focused answer:""",
        
        'procurement_officer': """You are an AI assistant for procurement officers managing procurement processes.
        Focus on process management, timelines, bidder coordination, operations, and administrative requirements.
        Provide actionable insights for managing procurement activities.
        
        Context: {context}
        Question: {question}
        
        Procurement management answer:""",
        
        'policy_maker': """You are an AI assistant for policy makers and decision makers.
        Focus on regulatory compliance, budget implications, strategic decisions, policy alignment, and governance.
        Provide strategic insights and policy considerations.
        
        Context: {context}
        Question: {question}
        
        Policy and strategic answer:""",
        
        'bidder': """You are an AI assistant helping bidders/suppliers understand procurement opportunities.
        Focus on specifications, submission requirements, deadlines, participation guidance, and competitive positioning.
        Provide clear guidance for successful participation.
        
        Context: {context}
        Question: {question}
        
        Bidder guidance answer:"""
    }
    
    # Create QA chains for each role
    for role, prompt_template in role_prompts.items():
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chains[role] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

def reset_progress():
    """Reset progress tracking for new session"""
    global processing_progress
    with progress_lock:
        processing_progress.update({
            'total_files': 0,
            'processed_files': 0,
            'current_file': '',
            'status': 'idle',
            'errors': [],
            'start_time': None,
            'folder_path': ''
        })

def update_progress(status=None, current_file=None, increment_processed=False, error=None):
    """Thread-safe progress update"""
    global processing_progress
    with progress_lock:
        if status:
            processing_progress['status'] = status
        if current_file:
            processing_progress['current_file'] = current_file
        if increment_processed:
            processing_progress['processed_files'] += 1
        if error:
            processing_progress['errors'].append(error)

def process_pdf_with_langchain(file_path, copy_to_uploads=False):
    """Process a PDF file using LangChain components"""
    try:
        update_progress(current_file=os.path.basename(file_path))
        
        # Copy file to uploads directory if requested
        if copy_to_uploads:
            try:
                filename = os.path.basename(file_path)
                target_path = os.path.join(UPLOAD_FOLDER, filename)
                
                if not os.path.exists(target_path):
                    shutil.copy2(file_path, target_path)
                    
            except Exception as e:
                print(f"Warning: Could not copy file to uploads directory: {e}")
        
        # Load PDF using LangChain PyPDFLoader
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            error_msg = f"No content extracted from {os.path.basename(file_path)}"
            update_progress(error=error_msg)
            return {"success": False, "file": file_path, "error": error_msg}
        
        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        
        # Add metadata to chunks
        for chunk in chunks:
            chunk.metadata.update({
                'source_file': os.path.basename(file_path),
                'full_path': file_path,
                'processed_at': datetime.now().isoformat()
            })
        
        return {
            "success": True,
            "file": file_path,
            "chunks": chunks,
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        error_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
        update_progress(error=error_msg)
        return {"success": False, "file": file_path, "error": error_msg}

def batch_index_documents(file_results):
    """Batch index all processed documents using ChromaDB"""
    
    # Ensure vectorstore is initialized
    vectorstore = get_or_initialize_vectorstore()
    if vectorstore is None:
        return {"success": False, "message": "Failed to initialize vector store"}
    
    try:
        # Collect all chunks from successful processing
        all_chunks = []
        for result in file_results:
            if result["success"]:
                all_chunks.extend(result["chunks"])
        
        if not all_chunks:
            return {"success": False, "message": "No valid chunks to index"}
        
        print(f"Adding {len(all_chunks)} chunks to ChromaDB...")
        
        # Add documents to ChromaDB
        vectorstore.add_documents(all_chunks)
        
        # ChromaDB persists automatically in newer versions
        
        return {
            "success": True,
            "total_chunks_added": len(all_chunks),
            "total_files_processed": sum(1 for r in file_results if r["success"]),
            "total_files_failed": sum(1 for r in file_results if not r["success"])
        }
        
    except Exception as e:
        return {"success": False, "message": f"Batch indexing failed: {str(e)}"}

def process_folder_contents(folder_path, max_workers=3):
    """Process all PDF files in a folder using LangChain"""
    global processing_progress
    
    try:
        reset_progress()
        
        # Find all PDF files
        pdf_files = []
        for extension in ['*.pdf', '*.PDF']:
            pdf_files.extend(glob.glob(os.path.join(folder_path, '**', extension), recursive=True))
        
        if not pdf_files:
            return {"success": False, "message": "No PDF files found in the specified folder"}
        
        # Initialize progress
        update_progress(status='processing')
        with progress_lock:
            processing_progress.update({
                'total_files': len(pdf_files),
                'processed_files': 0,
                'folder_path': folder_path,
                'start_time': time.time(),
                'errors': [],
                'current_file': ''
            })
        
        print(f"Found {len(pdf_files)} PDF files to process with LangChain")
        
        # Process files in parallel
        file_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_pdf_with_langchain, pdf_file, True): pdf_file 
                for pdf_file in pdf_files
            }
            
            for future in future_to_file:
                try:
                    result = future.result()
                    file_results.append(result)
                    update_progress(increment_processed=True)
                    
                    if processing_progress['status'] == 'cancelled':
                        break
                        
                except Exception as e:
                    pdf_file = future_to_file[future]
                    error_msg = f"Error processing {pdf_file}: {str(e)}"
                    update_progress(error=error_msg, increment_processed=True)
                    file_results.append({"success": False, "file": pdf_file, "error": error_msg})
        
        # Batch index all results
        print("Starting batch indexing with ChromaDB...")
        index_result = batch_index_documents(file_results)
        
        if index_result["success"]:
            update_progress(status='completed')
            return {
                "success": True,
                "message": f"Successfully processed {index_result['total_files_processed']} files",
                "total_files": len(pdf_files),
                "processed_files": index_result['total_files_processed'],
                "failed_files": index_result['total_files_failed'],
                "total_chunks": index_result['total_chunks_added'],
                "folder_path": folder_path,
                "errors": processing_progress['errors']
            }
        else:
            update_progress(status='error', error=index_result["message"])
            return {"success": False, "message": index_result["message"], "folder_path": folder_path}
            
    except Exception as e:
        update_progress(status='error', error=str(e))
        return {"success": False, "message": f"Folder processing failed: {str(e)}", "folder_path": folder_path}

# Flask routes

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/document-viewer.html')
def document_viewer():
    return send_from_directory('.', 'document-viewer.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/css/<filename>')
def css_files(filename):
    return send_from_directory('css', filename)

@app.route('/js/<filename>')
def js_files(filename):
    return send_from_directory('js', filename)

@app.route('/images/<filename>')
def image_files(filename):
    return send_from_directory('images', filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a single PDF file"""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "error": "Only PDF files are supported"})
        
        # Save uploaded file
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process with LangChain
        result = process_pdf_with_langchain(file_path, copy_to_uploads=False)
        
        if result["success"]:
            # Index the document
            index_result = batch_index_documents([result])
            
            if index_result["success"]:
                return jsonify({
                    "success": True,
                    "message": f"Successfully processed {filename}",
                    "filename": filename,
                    "chunks": result["chunks_count"],
                    "redirect": "document-viewer.html"
                })
            else:
                return jsonify({"success": False, "error": index_result["message"]})
        else:
            return jsonify({"success": False, "error": result["error"]})
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Upload failed: {str(e)}"})

@app.route('/upload-folder', methods=['POST'])
def upload_folder():
    """Process a folder of PDF files"""
    try:
        data = request.json
        if not data or 'folder_path' not in data:
            return jsonify({"success": False, "error": "No folder path provided"})
        
        folder_path = data['folder_path']
        max_workers = data.get('max_workers', 3)
        
        if not os.path.exists(folder_path):
            return jsonify({"success": False, "error": "Folder path does not exist"})
        
        # Start processing in background
        def process_async():
            result = process_folder_contents(folder_path, max_workers)
            return result
        
        # For now, process synchronously but we could make this async
        result = process_async()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": f"Folder upload failed: {str(e)}"})

@app.route('/query', methods=['POST'])
def query_documents():
    """Query documents using LangChain RAG"""
    try:
        # Ensure vectorstore is initialized
        vectorstore = get_or_initialize_vectorstore()
        if vectorstore is None:
            return jsonify({"error": "Vector store not initialized. Please upload documents first."}), 500
            
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query']
        role = data.get('role', 'general')
        
        # Get the appropriate QA chain for the role
        qa_chain = qa_chains.get(role, qa_chains['general'])
        
        # Run the query
        result = qa_chain({"query": query})
        
        # Extract source information
        sources = []
        relevant_files = set()
        
        if 'source_documents' in result:
            for doc in result['source_documents']:
                source_file = doc.metadata.get('source_file', 'unknown')
                relevant_files.add(source_file)
                sources.append({
                    'source_file': source_file,
                    'page': doc.metadata.get('page', 'N/A'),
                    'content_preview': doc.page_content[:200] + "..."
                })
        
        return jsonify({
            "response": result['result'],
            "role": role,
            "confirmed_role": role,
            "sources": sources,
            "relevant_files": list(relevant_files),
            "contexts_used": len(sources)
        })
        
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Check system status"""
    try:
        # Test Ollama connection
        test_response = llm.invoke("test")
        ollama_status = "connected"
    except:
        ollama_status = "disconnected"
    
    # Check ChromaDB status
    try:
        collection_count = vectorstore._collection.count()
        chroma_status = "connected"
    except:
        collection_count = 0
        chroma_status = "disconnected"
    
    return jsonify({
        "ollama_status": ollama_status,
        "chroma_status": chroma_status,
        "documents_indexed": collection_count,
        "system": "LangChain RAG System"
    })

@app.route('/upload-progress', methods=['GET'])
def get_upload_progress():
    """Get current upload progress"""
    return jsonify(processing_progress)

@app.route('/files', methods=['GET'])
def list_files():
    """Get list of indexed documents"""
    
    # Ensure vectorstore is initialized
    vectorstore = get_or_initialize_vectorstore()
    if vectorstore is None:
        return jsonify({"success": False, "error": "Vector store not initialized"})
    
    try:
        # Get unique source files from ChromaDB
        all_docs = vectorstore.get()
        
        file_stats = defaultdict(lambda: {
            'total_chunks': 0,
            'available_in_uploads': False
        })
        
        if all_docs and 'metadatas' in all_docs:
            for metadata in all_docs['metadatas']:
                source_file = metadata.get('source_file', 'unknown')
                file_stats[source_file]['total_chunks'] += 1
                
                # Check if file exists in uploads
                upload_path = os.path.join(UPLOAD_FOLDER, source_file)
                file_stats[source_file]['available_in_uploads'] = os.path.exists(upload_path)
        
        files_list = list(file_stats.keys())
        
        return jsonify({
            "success": True,
            "total_files": len(files_list),
            "files": files_list,
            "file_statistics": dict(file_stats)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/database-stats', methods=['GET'])
def get_database_stats():
    """Get database statistics"""
    # Ensure vectorstore is initialized
    vectorstore = get_or_initialize_vectorstore()
    if vectorstore is None:
        return jsonify({"success": False, "error": "Vector store not initialized"})
        
    try:
        collection = vectorstore._collection
        total_docs = collection.count()
        
        # Get sample of documents to analyze
        sample_docs = vectorstore.get(limit=1000)
        
        file_details = {}
        if sample_docs and 'metadatas' in sample_docs:
            for metadata in sample_docs['metadatas']:
                source_file = metadata.get('source_file', 'unknown')
                if source_file not in file_details:
                    filename = os.path.basename(source_file)
                    upload_path = os.path.join(UPLOAD_FOLDER, filename)
                    
                    file_details[source_file] = {
                        'filename': filename,
                        'available_in_uploads': os.path.exists(upload_path),
                        'upload_url': f'/uploads/{filename}' if os.path.exists(upload_path) else None
                    }
        
        unique_files = len(file_details)
        
        return jsonify({
            "success": True,
            "total_chunks": total_docs,
            "unique_files": unique_files,
            "files": list(file_details.keys()),
            "file_details": file_details
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    # Initialize LangChain components
    if initialize_langchain_components():
        print("Starting Flask application with LangChain RAG system...")
        app.run(host='127.0.0.1', port=5000, debug=True)
    else:
        print("Failed to initialize LangChain components. Please check your setup.")