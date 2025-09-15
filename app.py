# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import tempfile
import faiss
import numpy as np
import PyPDF2

# Using TF-IDF instead of sentence transformers for local embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from pathlib import Path
import re
from collections import defaultdict
import logging
from datetime import datetime

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Basic configuration
UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
VECTOR_DIMENSION = 1000
CHUNK_SIZE = 500  # Characters per text chunk
CHUNK_OVERLAP = 50

# Create folders if they don't exist
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(DB_FOLDER).mkdir(exist_ok=True)

# Local TF-IDF embeddings instead of external API
class LocalEmbeddings:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        
    def fit(self, texts):
        """Train the vectorizer on text corpus"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
    def encode(self, texts):
        """Convert texts to vectors"""
        if not self.is_fitted:
            self.fit(texts)
        
        if isinstance(texts, str):
            texts = [texts]
            
        vectors = self.vectorizer.transform(texts).toarray()
        return vectors.astype(np.float32)

model = LocalEmbeddings(max_features=VECTOR_DIMENSION)

# Vector database setup
faiss_index = faiss.IndexFlatIP(VECTOR_DIMENSION)  
document_chunks = []  # Store text chunks with metadata

import glob
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Global variable to track processing progress
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
        print(f"DEBUG: Progress reset - processed_files = {processing_progress['processed_files']}")

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
            print(f"DEBUG: Incremented counter to {processing_progress['processed_files']}")
        if error:
            processing_progress['errors'].append(error)

def process_single_file(file_path, copy_to_uploads=False):
    """Process a single file and return results"""
    try:
        update_progress(current_file=os.path.basename(file_path))
        
        # Copy file to uploads directory if requested (for folder uploads)
        uploaded_file_path = None
        if copy_to_uploads:
            try:
                filename = os.path.basename(file_path)
                target_path = os.path.join(UPLOAD_FOLDER, filename)
                
                # Check if the exact same file already exists
                if os.path.exists(target_path):
                    if os.path.getsize(file_path) == os.path.getsize(target_path):
                        print(f"File {filename} already exists with same size, skipping copy")
                        uploaded_file_path = target_path
                    else:
                        base_name, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(os.path.join(UPLOAD_FOLDER, filename)):
                            filename = f"{base_name}_{counter}{ext}"
                            counter += 1
                        
                        uploaded_file_path = os.path.join(UPLOAD_FOLDER, filename)
                        import shutil
                        shutil.copy2(file_path, uploaded_file_path)
                        print(f"Copied {file_path} to {uploaded_file_path}")
                else:
                    # File doesn't exist, copy it
                    uploaded_file_path = target_path
                    import shutil
                    shutil.copy2(file_path, uploaded_file_path)
                    print(f"Copied {file_path} to {uploaded_file_path}")
                    
            except Exception as e:
                print(f"Warning: Could not copy file to uploads directory: {e}")
                # Continue with processing even if copy fails
        
        # Extract chunks from file
        chunks = extract_pdf_text(file_path)
        if not chunks:
            error_msg = f"No text extracted from {os.path.basename(file_path)}"
            update_progress(error=error_msg)
            return {"success": False, "file": file_path, "error": error_msg, "chunks": 0}

        return {
            "success": True, 
            "file": file_path, 
            "uploaded_file": uploaded_file_path or file_path,
            "chunks": chunks, 
            "chunks_count": len(chunks)
        }
        
    except Exception as e:
        error_msg = f"Error processing {os.path.basename(file_path)}: {str(e)}"
        update_progress(error=error_msg)
        return {"success": False, "file": file_path, "error": error_msg, "chunks": 0}

def batch_index_files(file_results):
    """Efficiently batch index all processed files"""
    global content_indexer, model
    
    try:
        # Collect all chunks and texts
        all_new_chunks = []
        
        for result in file_results:
            if result["success"]:
                all_new_chunks.extend(result["chunks"])
        
        if not all_new_chunks:
            return {"success": False, "message": "No valid chunks to index"}
        
        # Prepare texts for model training
        new_texts = [chunk["text"] for chunk in all_new_chunks]
        existing_texts = [chunk["text"] for chunk in document_chunks]
        all_texts = existing_texts + new_texts
        
        # Retrain model on all data (existing + new)
        print(f"Retraining model on {len(all_texts)} total texts...")
        model.fit(all_texts)
        
        # Generate embeddings for all new chunks at once
        print(f"Generating embeddings for {len(new_texts)} new chunks...")
        embeddings = model.encode(new_texts)
        faiss.normalize_L2(embeddings)
        
        # Initialize content indexer if needed with correct dimension
        if embeddings.shape[1] != content_indexer.vector_dimension:
            print(f"Rebuilding content indexer with dimension {embeddings.shape[1]}")
            content_indexer = ContentSpecificIndexer(embeddings.shape[1])
        
        # Batch add all chunks to indexes
        print("Adding chunks to content-specific indexes...")
        for chunk, embedding in zip(all_new_chunks, embeddings):
            document_chunks.append(chunk)
            content_indexer.add_chunk(chunk, embedding)
        
        # Save everything once at the end
        save_index()
        
        return {
            "success": True, 
            "total_chunks_added": len(all_new_chunks),
            "total_files_processed": sum(1 for r in file_results if r["success"]),
            "total_files_failed": sum(1 for r in file_results if not r["success"])
        }
        
    except Exception as e:
        return {"success": False, "message": f"Batch indexing failed: {str(e)}"}

def process_folder_contents(folder_path, max_workers=3):
    """Process all PDF files in a folder efficiently"""
    global processing_progress
    
    try:
        # Reset progress tracking for new session
        reset_progress()
        
        # Find all PDF files in folder and subfolders
        pdf_files = []
        for extension in ['*.pdf', '*.PDF']:
            pdf_files.extend(glob.glob(os.path.join(folder_path, '**', extension), recursive=True))
        
        if not pdf_files:
            return {"success": False, "message": "No PDF files found in the specified folder"}
        
        # Initialize progress tracking
        update_progress(status='processing')
        with progress_lock:
            processing_progress.update({
                'total_files': len(pdf_files),
                'processed_files': 0,  # Explicitly reset counter
                'start_time': time.time(),
                'errors': [],
                'current_file': ''
            })
            print(f"DEBUG: Progress initialized - total={len(pdf_files)}, processed={processing_progress['processed_files']}")
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        # Process files in parallel (but limit concurrency to avoid memory issues)
        file_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file processing tasks - copy files to uploads directory for folder uploads
            future_to_file = {executor.submit(process_single_file, file_path, True): file_path 
                            for file_path in pdf_files}
            
            # Collect results as they complete
            for future in future_to_file:
                result = future.result()
                file_results.append(result)
                # Only increment for successful file processing, not pages
                if result["success"]:
                    update_progress(increment_processed=True)
                
                print(f"Processed {processing_progress['processed_files']}/{processing_progress['total_files']}: "
                      f"{os.path.basename(result['file'])} - {'Success' if result['success'] else 'Failed'}")
        
        # Batch index all successful results
        print("Starting batch indexing...")
        index_result = batch_index_files(file_results)
        
        # Update final status
        if index_result["success"]:
            update_progress(status='completed')
            return {
                "success": True,
                "message": f"Successfully processed {len(pdf_files)} files",
                "total_files": len(pdf_files),
                "successful_files": index_result["total_files_processed"],
                "failed_files": index_result["total_files_failed"],
                "total_chunks_added": index_result["total_chunks_added"],
                "errors": processing_progress['errors'],
                "folder_path": folder_path
            }
        else:
            update_progress(status='error')
            return {
                "success": False,
                "message": index_result["message"],
                "errors": processing_progress['errors'],
                "folder_path": folder_path
            }
            
    except Exception as e:
        update_progress(status='error', error=str(e))
        return {"success": False, "message": f"Folder processing failed: {str(e)}", "folder_path": folder_path}

# Enhanced indexing system with content-specific indexes
class ContentSpecificIndexer:
    def __init__(self, vector_dimension=1000):
        self.vector_dimension = vector_dimension
        self.indexes = {}
        self.chunk_mappings = {}
        self.is_initialized = False
        
    def initialize_indexes(self, actual_dimension):
        """Initialize indexes with the actual embedding dimension"""
        self.vector_dimension = actual_dimension
        self.indexes = {
            'CONTRACT_AMOUNT': faiss.IndexFlatIP(actual_dimension),
            'TIMELINE': faiss.IndexFlatIP(actual_dimension),
            'COMPLIANCE_REQUIREMENTS': faiss.IndexFlatIP(actual_dimension),
            'TECHNICAL_SPECS': faiss.IndexFlatIP(actual_dimension),
            'BIDDER_INFO': faiss.IndexFlatIP(actual_dimension),
            'CONTACT_INFO': faiss.IndexFlatIP(actual_dimension),
            'REFERENCE_INFO': faiss.IndexFlatIP(actual_dimension),
            'GENERAL': faiss.IndexFlatIP(actual_dimension)
        }
        self.chunk_mappings = {tag: [] for tag in self.indexes.keys()}
        self.is_initialized = True
        print(f"Initialized content-specific indexes with dimension: {actual_dimension}")
        
    def add_chunk(self, chunk, embedding):
        """Add chunk to appropriate content-specific indexes"""
        # Initialize indexes if not done yet
        if not self.is_initialized:
            self.initialize_indexes(embedding.shape[0])
        
        # Ensure embedding is 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        chunk_tags = chunk.get('metadata', {}).get('role_tags', ['GENERAL'])
        
        # Add to each relevant index based on tags
        for tag in chunk_tags:
            if tag in self.indexes:
                # Add embedding to specific index
                self.indexes[tag].add(embedding)
                # Track chunk mapping - store the global chunk index
                chunk_global_index = len(document_chunks) - 1  # Current index since chunk was just added
                self.chunk_mappings[tag].append(chunk_global_index)
                
    def search_targeted(self, query_embedding, target_tags, top_k=5):
        """Search only in relevant content-specific indexes"""
        all_results = []
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        for tag in target_tags:
            if tag in self.indexes and self.indexes[tag].ntotal > 0:
                try:
                    # Search in specific index
                    D, I = self.indexes[tag].search(query_embedding, min(top_k, self.indexes[tag].ntotal))
                    
                    # Map back to original chunk indices
                    for score, idx in zip(D[0], I[0]):
                        if idx < len(self.chunk_mappings[tag]):
                            original_idx = self.chunk_mappings[tag][idx]
                            all_results.append((score, original_idx, tag))
                except Exception as e:
                    print(f"Error searching in {tag} index: {e}")
                    continue
        
        # Sort by relevance score and return top results
        all_results.sort(key=lambda x: x[0], reverse=True)
        return all_results[:top_k]
    
    def rebuild_indexes(self, all_chunks, model):
        """Rebuild all indexes from scratch"""
        # Reset indexes
        for tag in self.indexes:
            self.indexes[tag] = faiss.IndexFlatIP(self.vector_dimension)
            self.chunk_mappings[tag] = []
        
        # Re-add all chunks
        if all_chunks:
            all_texts = [chunk["text"] for chunk in all_chunks]
            embeddings = model.encode(all_texts)
            faiss.normalize_L2(embeddings)
            
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
                self.add_chunk(chunk, embedding)

# Replace the simple FAISS index with content-specific indexer
content_indexer = ContentSpecificIndexer(VECTOR_DIMENSION)

# Load existing vector database
def load_index():
    global content_indexer, document_chunks, model
    if os.path.exists(f"{DB_FOLDER}/content_indexes.pkl") and os.path.exists(f"{DB_FOLDER}/chunks.json"):
        try:
            import pickle
            
            # Load chunk data
            with open(f"{DB_FOLDER}/chunks.json", 'r') as f:
                document_chunks = json.load(f)
            
            # Load content-specific indexes
            with open(f"{DB_FOLDER}/content_indexes.pkl", 'rb') as f:
                index_data = pickle.load(f)
                content_indexer.indexes = index_data['indexes']
                content_indexer.chunk_mappings = index_data['mappings']
            
            # Retrain model on existing data
            if document_chunks:
                texts = [chunk["text"] for chunk in document_chunks]
                model.fit(texts)
                print(f"Loaded content-specific indexes with {len(document_chunks)} chunks")
            
        except Exception as e:
            print(f"Error loading indexes: {e}")
            content_indexer = ContentSpecificIndexer(VECTOR_DIMENSION)
            document_chunks = []
    else:
        content_indexer = ContentSpecificIndexer(VECTOR_DIMENSION)
        document_chunks = []

# Save vector database
def save_index():
    import pickle
    
    # Save chunk data
    with open(f"{DB_FOLDER}/chunks.json", 'w') as f:
        json.dump(document_chunks, f)
    
    # Save content-specific indexes
    index_data = {
        'indexes': content_indexer.indexes,
        'mappings': content_indexer.chunk_mappings
    }
    with open(f"{DB_FOLDER}/content_indexes.pkl", 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"Saved content-specific indexes with {len(document_chunks)} chunks")

def auto_tag_chunk(text):
    """Tag chunks based on content keywords"""
    text_lower = text.lower()
    assigned_tags = []
    
    # Budget/cost detection
    if any(phrase in text for phrase in ["Approved Budget Cost", "Total Budget", "Contract Amount"]):
        assigned_tags.append("CONTRACT_AMOUNT")
    
    # Technical specifications
    if "abc:" in text_lower and any(item in text_lower for item in ["unit", "pcs", "piece", "specifications"]):
        assigned_tags.append("TECHNICAL_SPECS")
    
    # Timeline information
    if any(phrase in text_lower for phrase in ["closing", "deadline", "submission", "on or before", "date", "time"]):
        assigned_tags.append("TIMELINE")
    
    # Legal/compliance requirements
    if any(phrase in text_lower for phrase in ["republic act", "compliance", "criteria", "eligibility", "requirements", "regulation"]):
        assigned_tags.append("COMPLIANCE_REQUIREMENTS")
    
    # Bidder-related info
    if any(phrase in text_lower for phrase in ["bidder", "supplier", "quotation", "philgeps", "mayor's permit", "documents"]):
        assigned_tags.append("BIDDER_INFO")
    
    # Contact details
    if any(phrase in text_lower for phrase in ["phone", "telefax", "email", "@", "contact", "inquiries", "office"]):
        assigned_tags.append("CONTACT_INFO")
    
    # Reference numbers and IDs
    if any(phrase in text_lower for phrase in ["reference", "rfq", "pr-", "procurement", "number", "id"]):
        assigned_tags.append("REFERENCE_INFO")
    
    return assigned_tags if assigned_tags else ["GENERAL"]

def determine_chunk_priority(text, tags):
    """Set priority based on content importance"""
    text_lower = text.lower()
    
    # High priority content
    if any(phrase in text_lower for phrase in ["approved budget cost", "closing", "republic act", "request for quotation"]):
        return "high"
    
    # Low priority content
    if any(phrase in text_lower for phrase in ["contact", "phone", "email only"]):
        return "low"
    
    return "medium"

def determine_content_type(text, tags):
    """Classify content type"""
    text_lower = text.lower()
    
    if any(phrase in text_lower for phrase in ["request for quotation", "approved budget cost", "project"]):
        return "summary"
    
    if any(phrase in text_lower for phrase in ["republic act", "regulation", "criteria"]):
        return "regulatory"
    
    if any(phrase in text_lower for phrase in ["submission", "quotation", "documents"]):
        return "procedural"
    
    return "detailed"

def calculate_role_relevance(text, tags):
    """Score relevance for different user roles"""
    text_lower = text.lower()
    relevance = {
        "auditor": 0.5,
        "procurement_officer": 0.5, 
        "bidder": 0.5,
        "policy_maker": 0.5
    }
    
    # Boost relevance based on content type
    if "COMPLIANCE_REQUIREMENTS" in tags or "CONTRACT_AMOUNT" in tags:
        relevance["auditor"] += 0.4
    
    if "TIMELINE" in tags or "BIDDER_INFO" in tags:
        relevance["procurement_officer"] += 0.4
    
    if "TECHNICAL_SPECS" in tags or "BIDDER_INFO" in tags:
        relevance["bidder"] += 0.4
    
    if "COMPLIANCE_REQUIREMENTS" in tags or "CONTRACT_AMOUNT" in tags:
        relevance["policy_maker"] += 0.4
    
    # Cap at maximum
    for role in relevance:
        relevance[role] = min(1.0, relevance[role])
    
    return relevance

# Text chunking with metadata
def chunk_text(text, filename="", page_num=0):
    chunks = []
    i = 0
    while i < len(text):
        chunk_text = text[i:i + CHUNK_SIZE]
        if chunk_text:
            # Generate metadata for this chunk
            role_tags = auto_tag_chunk(chunk_text)
            chunk_priority = determine_chunk_priority(chunk_text, role_tags)
            content_type = determine_content_type(chunk_text, role_tags)
            role_relevance = calculate_role_relevance(chunk_text, role_tags)
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "start_char": i,
                    "end_char": min(i + CHUNK_SIZE, len(text)),
                    "role_tags": role_tags,
                    "role_relevance": role_relevance,
                    "chunk_priority": chunk_priority,
                    "content_type": content_type
                }
            })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# PDF text extraction
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

# Add document to vector database
def add_document_to_index(file_path):
    global content_indexer, model
    
    chunks = extract_pdf_text(file_path)
    if not chunks:
        return {"success": False, "message": "No text extracted from document"}
    
    # Prepare all texts for training
    texts = [chunk["text"] for chunk in chunks]
    all_existing_texts = [chunk["text"] for chunk in document_chunks]
    all_texts = all_existing_texts + texts
    
    # Retrain model on all data
    model.fit(all_texts)
    
    # Generate embeddings for new chunks
    embeddings = model.encode(texts)
    faiss.normalize_L2(embeddings)
    
    # Add to content-specific indexes
    for chunk, embedding in zip(chunks, embeddings):
        content_indexer.add_chunk(chunk, embedding)
        document_chunks.append(chunk)
    
    save_index()
    return {"success": True, "chunks_added": len(chunks)}

# Role-specific configurations
ROLE_TAGS = {
    'auditor': ['COMPLIANCE', 'BIDDER_INFO', 'BUDGET'],
    'procurement_officer': ['TIMELINE', 'BIDDER_INFO', 'BUDGET', 'SPECIFICATIONS'],
    'policy_maker': ['COMPLIANCE', 'BUDGET', 'TIMELINE'],
    'bidder': ['SPECIFICATIONS', 'BIDDER_INFO', 'TIMELINE', 'CONTACT_INFO']
}

ROLE_PROMPTS = {
    'auditor': "You are a procurement auditor. Focus on compliance, legal requirements, proper procedures, and budget verification in your analysis.",
    'procurement_officer': "You are a procurement officer. Focus on process management, timelines, bidder coordination, and ensuring smooth procurement operations.",
    'policy_maker': "You are a policy maker. Focus on regulatory compliance, budget implications, policy adherence, and strategic procurement decisions.",
    'bidder': "You are helping a bidder/supplier. Focus on specifications, submission requirements, deadlines, and what bidders need to know to participate successfully."
}

# Keyword mapping for content detection
STRICT_TAG_KEYWORDS = {
    'CONTRACT_AMOUNT': ['budget', 'cost', 'amount', 'price', 'php', 'contract amount', 'approved budget', 'total cost', 'total budget'],
    'TIMELINE': ['deadline', 'closing', 'submission', 'date', 'time', 'schedule', 'timeline', 'when', 'due date'],
    'COMPLIANCE_REQUIREMENTS': ['compliance', 'requirements', 'legal', 'republic act', 'regulation', 'criteria', 'eligibility', 'act', 'law'],
    'TECHNICAL_SPECS': ['specifications', 'specs', 'dimensions', 'materials', 'technical', 'description', 'features', 'length', 'width', 'height', 'inches', 'size', 'table', 'chair', 'truck', 'testing table', 'office table'],
    'BIDDER_INFO': ['bidder', 'supplier', 'quotation', 'submission', 'documents', 'eligibility', 'permit', 'philgeps'],
    'CONTACT_INFO': ['contact', 'phone', 'email', 'address', 'office', 'inquiries', 'telephone', 'secretariat'],
    'REFERENCE_INFO': ['reference', 'reference number', 'procurement id', 'id', 'number', 'rfq', 'pr number', 'document number', 'identification']
}

ROLE_CONTEXTS = {
    'auditor': {
        'framing': 'From an audit perspective, focusing on compliance and verification:',
        'priority_tags': ['COMPLIANCE_REQUIREMENTS', 'CONTRACT_AMOUNT'],
        'emphasis': 'regulatory compliance and proper documentation'
    },
    'procurement_officer': {
        'framing': 'From a procurement management perspective, focusing on operations:',
        'priority_tags': ['TIMELINE', 'BIDDER_INFO', 'TECHNICAL_SPECIFICATIONS'],
        'emphasis': 'process efficiency and stakeholder coordination'
    },
    'policy_maker': {
        'framing': 'From a strategic policy perspective, focusing on high-level decisions:',
        'priority_tags': ['COMPLIANCE_REQUIREMENTS', 'CONTRACT_AMOUNT', 'TIMELINE'],
        'emphasis': 'strategic implications and policy adherence'
    },
    'bidder': {
        'framing': 'For bidder guidance, focusing on participation requirements:',
        'priority_tags': ['TECHNICAL_SPECIFICATIONS', 'BIDDER_INFO', 'TIMELINE'],
        'emphasis': 'submission requirements and competitive advantage'
    }
}

def detect_query_intent(query):
    """Figure out what the user is asking about"""
    query_lower = query.lower()
    
    # Score each category based on keyword matches
    tag_scores = defaultdict(int)
    
    for tag, keywords in STRICT_TAG_KEYWORDS.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact phrase match gets highest score
            if keyword_lower in query_lower:
                # Check for word boundaries to avoid partial matches
                import re
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                if re.search(pattern, query_lower):
                    tag_scores[tag] += 5
                else:
                    tag_scores[tag] += 2
            
            # Additional synonym matches
            if tag == 'CONTRACT_AMOUNT' and any(word in query_lower for word in ['cost', 'money', 'price', 'budget', 'amount']):
                tag_scores[tag] += 1
            elif tag == 'TIMELINE' and any(word in query_lower for word in ['when', 'deadline', 'time', 'date', 'schedule']):
                tag_scores[tag] += 1
            elif tag == 'TECHNICAL_SPECS' and any(word in query_lower for word in ['spec', 'description', 'feature', 'requirement']):
                tag_scores[tag] += 1
            elif tag == 'COMPLIANCE_REQUIREMENTS' and any(word in query_lower for word in ['rule', 'law', 'must', 'required', 'comply']):
                tag_scores[tag] += 1
            elif tag == 'BIDDER_INFO' and any(word in query_lower for word in ['how to', 'submit', 'apply', 'participate']):
                tag_scores[tag] += 1
            elif tag == 'CONTACT_INFO' and any(word in query_lower for word in ['who', 'contact', 'reach', 'phone', 'email']):
                tag_scores[tag] += 1
    
    # Return best match if confident enough
    if tag_scores:
        best_tag = max(tag_scores, key=tag_scores.get)
        best_score = tag_scores[best_tag]
        
        print(f"DEBUG: Intent detection - Query: '{query}', Scores: {dict(tag_scores)}, Best: {best_tag}({best_score})")
        
        if best_score >= 2:  # Minimum confidence threshold
            return best_tag, best_score
    
    print(f"DEBUG: Intent detection - No clear intent found for query: '{query}'")
    return None, 0

def apply_strict_tag_filtering(chunks, required_tag, fallback_allowed=True):
    """Filter chunks by specific tag"""
    if not required_tag:
        return chunks
    
    filtered_chunks = []
    
    for chunk in chunks:
        chunk_tags = chunk.get('metadata', {}).get('role_tags', [])
        if required_tag in chunk_tags:
            filtered_chunks.append(chunk)
    
    # Fall back to all chunks if nothing found
    if not filtered_chunks and fallback_allowed:
        return chunks
    
    return filtered_chunks

def prioritize_chunks(chunks, role):
    """Keep for backward compatibility"""
    return prioritize_chunks_enhanced(chunks, role)

def enhanced_chunk_filtering(query, role, all_chunks, top_k=5):
    """Smart filtering using content-specific indexes for targeted search"""
    
    print(f"DEBUG: enhanced_chunk_filtering with targeted indexing - role: '{role}', query: '{query}'")
    role_logger.info(f"FILTERING_START: Role='{role}', Query='{query}', Total_chunks={len(all_chunks)}")
    
    # Detect what the user is asking about
    required_tag, confidence = detect_query_intent(query)
    print(f"DEBUG: Detected intent: {required_tag} with confidence: {confidence}")
    
    # Determine which content indexes to search
    target_tags = []
    if required_tag and confidence >= 2:
        # High confidence - search only specific content type
        target_tags = [required_tag]
        print(f"DEBUG: High confidence search - targeting only: {target_tags}")
    else:
        # Lower confidence - search role-relevant content types
        role_context = ROLE_CONTEXTS.get(role, {})
        target_tags = role_context.get('priority_tags', ['GENERAL'])
        print(f"DEBUG: Role-based search - targeting: {target_tags}")
    
    # Perform targeted search using content-specific indexes
    candidate_chunks = []
    if model.is_fitted and len(all_chunks) > 0:
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search only relevant content indexes
        search_results = content_indexer.search_targeted(query_embedding, target_tags, top_k * 2)
        
        # Convert results back to chunks
        for score, chunk_idx, content_tag in search_results:
            if chunk_idx < len(all_chunks):
                chunk = all_chunks[chunk_idx].copy()
                chunk['_search_score'] = score
                chunk['_matched_tag'] = content_tag
                candidate_chunks.append(chunk)
        
        print(f"DEBUG: Targeted search found {len(candidate_chunks)} candidates from {len(target_tags)} content types")
    else:
        # Fallback to tag-based filtering
        candidate_chunks = apply_strict_tag_filtering(all_chunks, required_tag, fallback_allowed=True)[:top_k * 2]
    
    # Apply role-specific prioritization
    prioritized_chunks = prioritize_chunks_enhanced(candidate_chunks, role)
    final_chunks = prioritized_chunks[:top_k]
    
    # Get role context for response framing
    role_context = ROLE_CONTEXTS.get(role, {})
    
    # Log results with targeted search info
    role_logger.info(f"TARGETED_SEARCH: Role='{role}', Intent='{required_tag}', Target_tags={target_tags}, Results={len(final_chunks)}")
    
    return {
        'chunks': final_chunks,
        'detected_intent': required_tag,
        'intent_confidence': confidence,
        'target_tags': target_tags,
        'targeted_search': True,
        'role_framing': role_context.get('framing', ''),
        'role_emphasis': role_context.get('emphasis', ''),
        'total_candidates': len(candidate_chunks),
        'filtering_applied': required_tag is not None
    }

def prioritize_chunks_enhanced(chunks, role):
    """Score and sort chunks based on role preferences"""
    if not chunks:
        return chunks
    
    print(f"DEBUG: prioritize_chunks_enhanced called with role: '{role}', chunks: {len(chunks)}")
    
    # Get role-specific priority tags
    role_context = ROLE_CONTEXTS.get(role, {})
    priority_tags = role_context.get('priority_tags', [])
    
    print(f"DEBUG: Priority tags for role '{role}': {priority_tags}")
    
    def chunk_score(chunk):
        metadata = chunk.get('metadata', {})
        score = 0
        
        # Base score from chunk priority
        priority_map = {'high': 3, 'medium': 2, 'low': 1}
        chunk_priority = metadata.get('chunk_priority', 'medium')
        score += priority_map.get(chunk_priority, 2)
        
        # Content type preference
        content_type = metadata.get('content_type', 'detailed')
        if role == 'policy_maker' and content_type in ['summary', 'regulatory']:
            score += 2
        elif role == 'auditor' and content_type in ['regulatory', 'procedural']:
            score += 2
        elif content_type == 'summary':
            score += 1
        
        # Role-specific tag matching
        chunk_tags = metadata.get('role_tags', [])
        tag_bonus = 0
        for tag in chunk_tags:
            if tag in priority_tags:
                tag_bonus += 2  # Bonus for matching priority tags
        score += tag_bonus
        
        # Role relevance score
        role_relevance = metadata.get('role_relevance', {}).get(role, 0.5)
        relevance_bonus = role_relevance * 3
        score += relevance_bonus
        
        print(f"DEBUG: Chunk scoring - Tags: {chunk_tags}, Priority_tags: {priority_tags}, Tag_bonus: {tag_bonus}, Role_relevance: {role_relevance}, Total_score: {score}")
        
        return score
    
    # Sort by score
    scored_chunks = sorted(chunks, key=chunk_score, reverse=True)
    
    # Debug top scores
    for i, chunk in enumerate(scored_chunks[:3]):
        score = chunk_score(chunk)
        tags = chunk.get('metadata', {}).get('role_tags', [])
        relevance = chunk.get('metadata', {}).get('role_relevance', {}).get(role, 0.5)
        print(f"DEBUG: Top chunk {i+1}: Score={score:.2f}, Tags={tags}, Relevance={relevance:.2f}")
    
    return scored_chunks

def create_fallback_response(query, role):
    """Generate helpful response when no relevant chunks found"""
    role_context = ROLE_CONTEXTS.get(role, {})
    
    fallback_message = f"""I couldn't find specific information to answer your query: "{query}".

{role_context.get('framing', 'From your perspective,')} I recommend:

1. Checking if your question relates to information that might be in other procurement documents
2. Contacting the BAC Secretariat at (062) 991-1771 loc 1003 or bac@wmsu.edu.ph for clarification
3. Reviewing the complete procurement documents for comprehensive information

Please rephrase your question or ask about specific aspects like budget, timeline, specifications, or compliance requirements."""
    
    return fallback_message

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('role_awareness.log'),
        logging.StreamHandler()
    ]
)

role_logger = logging.getLogger('role_awareness')

def log_query_result(query, role, detected_tag, confidence, chunks_used, role_relevance_scores, tags_used):
    """Log detailed query processing results"""
    
    role_logger.info("="*80)
    role_logger.info(f"🔍 NEW QUERY PROCESSED")
    role_logger.info(f"QUERY: '{query}'")
    role_logger.info(f"ROLE: {role.upper()}")
    role_logger.info(f"DETECTED_TAG: {detected_tag}")
    role_logger.info(f"CONFIDENCE: {confidence}")
    role_logger.info(f"CHUNKS_USED: {chunks_used}")
    
    if tags_used:
        role_logger.info(f"TAGS_IN_RESULTS: {', '.join(tags_used)}")
    
    if role_relevance_scores:
        avg_relevance = sum(role_relevance_scores) / len(role_relevance_scores)
        role_logger.info(f"ROLE_EFFECTIVENESS: {role.upper()} | AVG_RELEVANCE: {avg_relevance:.3f}")
        role_logger.info(f"INDIVIDUAL_RELEVANCE_SCORES: {[f'{s:.3f}' for s in role_relevance_scores]}")
    
    role_logger.info("="*80)

# Basic Ollama query function
def query_ollama(query, top_k=3):
    # Get query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search for relevant chunks
    D, I = faiss_index.search(query_embedding, top_k)
    
    if len(I[0]) == 0:
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Build context from relevant chunks
    contexts = []
    for idx in I[0]:
        if idx < len(document_chunks):
            contexts.append(document_chunks[idx]["text"])
    
    # Create prompt with context
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
                    "num_gpu": 0  # Use CPU
                }
            }
        )
        
        if response.status_code == 200:
            return {"response": response.json()["response"]}
        else:
            return {"error": f"Ollama error: {response.text}"}
    except Exception as e:
        return {"error": f"Error querying Ollama: {str(e)}"}

# Role-aware query function
def query_ollama_with_role(query, role='general', top_k=5):
    # Get similar chunks
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    D, I = faiss_index.search(query_embedding, top_k * 2)  # Get extra to filter
    
    if len(I[0]) == 0:
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Score chunks by role relevance
    contexts = []
    role_tags = ROLE_TAGS.get(role, [])
    
    for idx in I[0]:
        if idx < len(document_chunks):
            chunk = document_chunks[idx]
            
            # Calculate relevance score
            relevance_score = 1.0
            if 'role_tags' in chunk.get('metadata', {}):
                chunk_tags = chunk['metadata']['role_tags']
                # Boost for matching tags
                tag_match = len(set(chunk_tags) & set(role_tags))
                if tag_match > 0:
                    relevance_score = 1.0 + (tag_match * 0.3)
            
            if 'role_relevance' in chunk.get('metadata', {}):
                role_relevance = chunk['metadata']['role_relevance'].get(role, 0.5)
                relevance_score *= role_relevance
            
            contexts.append({
                'text': chunk['text'],
                'score': relevance_score,
                'tags': chunk.get('metadata', {}).get('role_tags', []),
                'source': chunk.get('metadata', {}).get('source', 'unknown')
            })
    
    # Sort by relevance and take top results
    contexts.sort(key=lambda x: x['score'], reverse=True)
    contexts = contexts[:top_k]
    
    # Build role-specific prompt
    context_text = "\n\n".join([ctx['text'] for ctx in contexts])
    role_prompt = ROLE_PROMPTS.get(role, "You are an expert in procurement documents.")
    
    # Add role-specific emphasis
    emphasis_tags = ", ".join([f"[{tag}]" for tag in role_tags])
    if emphasis_tags:
        role_prompt += f" Pay special attention to information tagged with: {emphasis_tags}."
    
    prompt = f"""
    {role_prompt}
    
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
                    "num_gpu": 0
                }
            }
        )
        
        if response.status_code == 200:
            return {
                "response": response.json()["response"],
                "role": role,
                "contexts_used": len(contexts),
                "relevant_tags": list(set([tag for ctx in contexts for tag in ctx['tags']]))
            }
        else:
            return {"error": f"Ollama error: {response.text}"}
    except Exception as e:
        return {"error": f"Error querying Ollama: {str(e)}"}

# Main query function with enhanced filtering
def query_ollama_with_strict_filtering(query, role='general', top_k=5):
    """Main query function with targeted content-specific indexing"""
    
    print(f"DEBUG: query_ollama_with_strict_filtering with targeted indexing - role: '{role}', query: '{query}'")
    role_logger.info(f"QUERY_START: Role='{role}', Query='{query}', Available_chunks={len(document_chunks)}")
    
    if len(document_chunks) == 0:
        log_query_result(query, role, None, 0, 0, [], [])
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Apply enhanced filtering with targeted indexing
    filter_result = enhanced_chunk_filtering(query, role, document_chunks, top_k)
    
    chunks = filter_result['chunks']

    # Extract role relevance scores and metadata
    role_relevance_scores = []
    all_tags_used = set()
    matched_content_types = set()
    
    for chunk in chunks:
        relevance = chunk.get('metadata', {}).get('role_relevance', {}).get(role, 0.5)
        role_relevance_scores.append(relevance)
        
        chunk_tags = chunk.get('metadata', {}).get('role_tags', [])
        all_tags_used.update(chunk_tags)
        
        # Track which content types were matched
        if '_matched_tag' in chunk:
            matched_content_types.add(chunk['_matched_tag'])

    tags_used_list = list(all_tags_used)
    matched_types_list = list(matched_content_types)

    print(f"DEBUG: Targeted search matched content types: {matched_types_list}")
    print(f"DEBUG: Role relevance scores: {role_relevance_scores}")
    print(f"DEBUG: Selected {len(chunks)} chunks from targeted search")

    # Handle no results case
    if not chunks:
        log_query_result(query, role, filter_result.get('detected_intent'), 0, 0, [], [])
        fallback_response = create_fallback_response(query, role)
        return {
            "response": fallback_response,
            "role": role,
            "contexts_used": 0,
            "filtering_info": filter_result,
            "is_fallback": True
        }
    
    # Build context with targeted content emphasis
    contexts_text = []
    for chunk in chunks:
        contexts_text.append(chunk['text'])
    
    context_text = "\n\n".join(contexts_text)
    
    # Enhanced prompt with targeted search context
    role_framing = filter_result.get('role_framing', '')
    detected_intent = filter_result.get('detected_intent', '')
    target_tags = filter_result.get('target_tags', [])
    
    # Role-specific instructions
    role_specific_instruction = ""
    if role == 'auditor':
        role_specific_instruction = "Focus on compliance, legal requirements, budget verification, and procedural correctness. Highlight any potential issues or red flags."
    elif role == 'procurement_officer':
        role_specific_instruction = "Focus on timeline management, process efficiency, bidder coordination, and operational aspects. Provide actionable information for managing the procurement process."
    elif role == 'policy_maker':
        role_specific_instruction = "Focus on strategic implications, policy compliance, budget impact, and high-level decision-making factors. Provide insights for strategic planning."
    elif role == 'bidder':
        role_specific_instruction = "Focus on submission requirements, specifications, deadlines, and what bidders need to know to participate successfully. Provide clear, actionable guidance."
    
    prompt = f"""
    {role_framing}
    
    {role_specific_instruction}
    
    {f"This response is based on targeted search for: {', '.join(target_tags)}" if target_tags else ""}
    {f"The query appears to be about: {detected_intent}" if detected_intent else ""}
    
    Context information (from targeted content search):
    {context_text}
    
    Query: {query}
    
    Provide a clear, role-appropriate answer based only on the provided context. The context has been specifically selected for relevance to your query, so focus on the most pertinent information.
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
                    "num_gpu": 0
                }
            }
        )
        
        if response.status_code == 200:
            # Log successful targeted query
            log_query_result(query, role, detected_intent, filter_result.get('intent_confidence', 0), len(chunks), role_relevance_scores, tags_used_list)
            
            # Log targeted search effectiveness
            role_logger.info(f"TARGETED_SEARCH_ANALYSIS for query '{query}' with role '{role}':")
            role_logger.info(f"  Target_content_types: {target_tags}")
            role_logger.info(f"  Matched_content_types: {matched_types_list}")
            role_logger.info(f"  Chunks_from_targeted_search: {len(chunks)}")
            
            avg_relevance = sum(role_relevance_scores) / len(role_relevance_scores) if role_relevance_scores else 0.5
            
            response_data = {
                "response": response.json()["response"],
                "role": role,
                "confirmed_role": role,
                "contexts_used": len(chunks),
                "relevant_tags": tags_used_list,
                "matched_content_types": matched_types_list,
                "target_content_types": target_tags,
                "sources": list(set([chunk.get('metadata', {}).get('source', 'unknown') for chunk in chunks])),
                "filtering_info": filter_result,
                "is_fallback": False,
                "targeted_search": True,
                "role_effectiveness": {
                    "avg_relevance": avg_relevance,
                    "tags_found": len(tags_used_list),
                    "intent_detected": detected_intent,
                    "search_precision": len(matched_types_list) / max(1, len(target_tags))
                }
            }
            
            return response_data
        else:
            role_logger.error(f"Ollama error: {response.text}")
            return {"error": f"Ollama error: {response.text}"}
    except Exception as e:
        role_logger.error(f"Error querying Ollama: {str(e)}")
        return {"error": f"Error querying Ollama: {str(e)}"}

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    """Serve uploaded PDF files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status - mainly for Ollama connection check"""
    return jsonify({
        "success": True,
        "ollama_status": "connected",  # Simplified for now
        "message": "System operational"
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload endpoint called!")
    
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        print(f"Invalid file type: {file.filename}")
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    print(f"Processing file: {file.filename}")
    
    # Save and process file
    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)
    print(f"File saved to: {temp_path}")
    
    result = add_document_to_index(temp_path)
    print(f"Processing result: {result}")
    
    return jsonify(result)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400

    # Extract and validate role
    role = data.get('role', 'general')
    
    # Debug logging
    print(f"DEBUG: Received request data: {data}")
    print(f"DEBUG: Extracted role: '{role}' (type: {type(role)})")
    
    # Clean up role string
    if isinstance(role, str):
        role = role.strip().lower()
    
    # Validate role
    role_mapping = {
        'general': 'general',
        'auditor': 'auditor',
        'procurement_officer': 'procurement_officer',
        'policy_maker': 'policy_maker',
        'bidder': 'bidder'
    }
    
    if role in role_mapping:
        role = role_mapping[role]
    else:
        print(f"DEBUG: Invalid role '{role}', defaulting to 'general'")
        role_logger.warning(f"Invalid role '{role}', defaulting to 'general'")
        role = 'general'
    
    print(f"DEBUG: Final role being used: '{role}'")
    
    # Log role switch
    role_logger.info("🔄" + "="*50 + "🔄")
    role_logger.info(f"🎭 ROLE SWITCH: Now operating as '{role.upper()}'")
    role_logger.info(f"📝 Query: '{data['query']}'")
    role_logger.info("🔄" + "="*50 + "🔄")
    
    role_logger.info(f"RECEIVED_ROLE: '{role}' for query: '{data['query']}'")
    
    # Process query with confirmed role
    result = query_ollama_with_strict_filtering(data['query'], role)
    
    # Clean up response
    clean_result = {
        "response": result.get("response", ""),
        "role": role,
        "confirmed_role": role,
        "contexts_used": result.get("contexts_used", 0),
        "relevant_tags": result.get("relevant_tags", []),
        "is_fallback": result.get("is_fallback", False),
        "filtering_applied": result.get("filtering_info", {}).get("filtering_applied", False),
        "detected_intent": result.get("filtering_info", {}).get("detected_intent", None)
    }
    
    print(f"DEBUG: Response role confirmation: '{role}'")
    role_logger.info(f"FINAL_RESPONSE_ROLE: '{role}' | TAGS: {clean_result.get('relevant_tags', [])} | CHUNKS: {clean_result.get('contexts_used', 0)}")
    
    return jsonify(clean_result)

@app.route('/get-roles', methods=['GET'])
def get_roles():
    """Return available roles for the frontend"""
    return jsonify({
        "roles": list(ROLE_TAGS.keys()),
        "role_descriptions": ROLE_PROMPTS
    })

@app.route('/analyze-chunks', methods=['GET'])
def analyze_chunks():
    """Analyze current chunks for debugging"""
    analysis = {
        "total_chunks": len(document_chunks),
        "chunks_with_role_tags": 0,
        "tag_distribution": {},
        "role_relevance_stats": {}
    }
    
    for chunk in document_chunks:
        metadata = chunk.get('metadata', {})
        
        if 'role_tags' in metadata:
            analysis["chunks_with_role_tags"] += 1
            for tag in metadata['role_tags']:
                analysis["tag_distribution"][tag] = analysis["tag_distribution"].get(tag, 0) + 1
        
        if 'role_relevance' in metadata:
            for role, score in metadata['role_relevance'].items():
                if role not in analysis["role_relevance_stats"]:
                    analysis["role_relevance_stats"][role] = {"scores": [], "avg": 0}
                analysis["role_relevance_stats"][role]["scores"].append(score)
    
    # Calculate averages
    for role, stats in analysis["role_relevance_stats"].items():
        if stats["scores"]:
            stats["avg"] = sum(stats["scores"]) / len(stats["scores"])
    
    return jsonify(analysis)

@app.route('/analyze-tagging', methods=['GET'])
def analyze_tagging():
    """Analyze tagging consistency across documents"""
    analysis = {
        "total_chunks": len(document_chunks),
        "documents": {},
        "tag_distribution": {},
        "role_coverage": {}
    }
    
    # Group by document
    for chunk in document_chunks:
        metadata = chunk.get('metadata', {})
        source = metadata.get('source', 'unknown')
        
        if source not in analysis["documents"]:
            analysis["documents"][source] = {
                "chunk_count": 0,
                "tags": set(),
                "has_auto_tags": False
            }
        
        analysis["documents"][source]["chunk_count"] += 1
        
        # Check for role tags
        role_tags = metadata.get('role_tags', [])
        if role_tags:
            analysis["documents"][source]["has_auto_tags"] = True
            analysis["documents"][source]["tags"].update(role_tags)
            
            # Update global tag distribution
            for tag in role_tags:
                analysis["tag_distribution"][tag] = analysis["tag_distribution"].get(tag, 0) + 1
    
    # Convert sets to lists for JSON
    for doc_info in analysis["documents"].values():
        doc_info["tags"] = list(doc_info["tags"])
    
    # Check role coverage
    for role in ["auditor", "procurement_officer", "bidder", "policy_maker"]:
        chunks_for_role = 0
        for chunk in document_chunks:
            role_relevance = chunk.get('metadata', {}).get('role_relevance', {})
            if role_relevance.get(role, 0) > 0.7:  # High relevance threshold
                chunks_for_role += 1
        analysis["role_coverage"][role] = chunks_for_role
    
    return jsonify(analysis)

@app.route('/analyze-index-efficiency', methods=['GET'])
def analyze_index_efficiency():
    """Analyze the efficiency of content-specific indexing"""
    analysis = {
        "total_chunks": len(document_chunks),
        "content_index_distribution": {},
        "index_sizes": {},
        "search_efficiency": {}
    }
    
    # Analyze distribution across content indexes
    for tag, chunks in content_indexer.chunk_mappings.items():
        analysis["content_index_distribution"][tag] = len(chunks)
        analysis["index_sizes"][tag] = content_indexer.indexes[tag].ntotal
    
    # Calculate search efficiency metrics
    total_indexed = sum(analysis["index_sizes"].values())
    for tag, size in analysis["index_sizes"].items():
        if total_indexed > 0:
            analysis["search_efficiency"][tag] = {
                "percentage_of_total": (size / total_indexed) * 100,
                "search_reduction": ((total_indexed - size) / total_indexed) * 100 if size < total_indexed else 0
            }
    
    return jsonify(analysis)

def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

# Initialize on startup
load_index()

@app.route('/upload-folder', methods=['POST'])
def upload_folder():
    """Handle folder path upload and process all PDFs in it"""
    data = request.json
    
    if not data or 'folder_path' not in data:
        return jsonify({"error": "No folder path provided"}), 400
    
    folder_path = data['folder_path']
    
    # Validate folder path
    if not os.path.exists(folder_path):
        return jsonify({"error": "Folder path does not exist"}), 400
    
    if not os.path.isdir(folder_path):
        return jsonify({"error": "Path is not a directory"}), 400
    
    print(f"Processing folder: {folder_path}")
    
    # Get optional parameters
    max_workers = data.get('max_workers', 3)  # Limit concurrent processing
    
    # Process folder in background (for large folders)
    def process_in_background():
        # Set folder path in progress tracking
        global processing_progress
        with progress_lock:
            processing_progress['folder_path'] = folder_path
        
        result = process_folder_contents(folder_path, max_workers)
        print(f"Folder processing completed: {result}")
    
    # Start background processing
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "success": True,
        "message": f"Started processing folder: {folder_path}",
        "status": "processing_started"
    })

@app.route('/upload-progress', methods=['GET'])
def get_upload_progress():
    """Get current upload/processing progress"""
    global processing_progress
    with progress_lock:
        progress_copy = processing_progress.copy()
        
        # Calculate elapsed time and ETA
        if progress_copy['start_time'] and progress_copy['status'] == 'processing':
            elapsed = time.time() - progress_copy['start_time']
            if progress_copy['processed_files'] > 0:
                avg_time_per_file = elapsed / progress_copy['processed_files']
                remaining_files = progress_copy['total_files'] - progress_copy['processed_files']
                eta_seconds = avg_time_per_file * remaining_files
                progress_copy['eta_seconds'] = eta_seconds
                progress_copy['elapsed_seconds'] = elapsed
            
        return jsonify(progress_copy)

@app.route('/database-stats', methods=['GET'])
def get_database_stats():
    """Get current database statistics"""
    try:
        chunks_file = f"{DB_FOLDER}/chunks.json"
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r') as f:
                chunks = json.load(f)
            
            # Count unique source files and build file info
            unique_sources = set()
            file_info = {}
            
            for chunk in chunks:
                source = chunk.get('metadata', {}).get('source', 'unknown')
                unique_sources.add(source)
                
                if source not in file_info:
                    # Check if file exists in uploads directory
                    filename = os.path.basename(source)
                    upload_path = os.path.join(UPLOAD_FOLDER, filename)
                    
                    file_info[source] = {
                        'original_path': source,
                        'filename': filename,
                        'available_in_uploads': os.path.exists(upload_path),
                        'upload_url': f'/uploads/{filename}' if os.path.exists(upload_path) else None
                    }
            
            return jsonify({
                "success": True,
                "total_chunks": len(chunks),
                "unique_files": len(unique_sources),
                "files": list(unique_sources),
                "file_details": file_info
            })
        else:
            return jsonify({
                "success": True,
                "total_chunks": 0,
                "unique_files": 0,
                "files": [],
                "file_details": {}
            })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/cancel-processing', methods=['POST'])
def cancel_processing():
    """Cancel ongoing folder processing"""
    global processing_progress
    with progress_lock:
        if processing_progress['status'] == 'processing':
            processing_progress['status'] = 'cancelled'
            return jsonify({"success": True, "message": "Processing cancelled"})
        else:
            return jsonify({"success": False, "message": "No active processing to cancel"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)