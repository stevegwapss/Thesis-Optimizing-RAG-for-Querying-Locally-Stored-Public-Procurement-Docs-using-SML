# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import tempfile
import faiss
import numpy as np
import PyPDF2

# Remove sentence_transformers import
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from pathlib import Path
import re
from collections import defaultdict
import logging
from datetime import datetime

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
VECTOR_DIMENSION = 1000  # Changed for TF-IDF
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(DB_FOLDER).mkdir(exist_ok=True)

# Initialize local embedding model (TF-IDF)
class LocalEmbeddings:
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        
    def fit(self, texts):
        """Fit the vectorizer on the corpus"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
    def encode(self, texts):
        """Encode texts to vectors"""
        if not self.is_fitted:
            # Fit on the provided texts if not already fitted
            self.fit(texts)
        
        if isinstance(texts, str):
            texts = [texts]
            
        vectors = self.vectorizer.transform(texts).toarray()
        return vectors.astype(np.float32)

model = LocalEmbeddings(max_features=VECTOR_DIMENSION)

# Initialize FAISS index
faiss_index = faiss.IndexFlatIP(VECTOR_DIMENSION)  # Changed to Inner Product for TF-IDF
document_chunks = []  # Store text chunks corresponding to vectors

# Load existing index if available
def load_index():
    global faiss_index, document_chunks, model
    if os.path.exists(f"{DB_FOLDER}/index.faiss") and os.path.exists(f"{DB_FOLDER}/chunks.json"):
        try:
            faiss_index = faiss.read_index(f"{DB_FOLDER}/index.faiss")
            with open(f"{DB_FOLDER}/chunks.json", 'r') as f:
                document_chunks = json.load(f)
            
            # If we have existing chunks, refit the model
            if document_chunks:
                texts = [chunk["text"] for chunk in document_chunks]
                model.fit(texts)
                print(f"Refitted model on {len(texts)} existing chunks")
            
            print(f"Loaded existing index with {len(document_chunks)} chunks")
        except Exception as e:
            print(f"Error loading index: {e}")
            # Initialize new index - fix the dimension issue
            faiss_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
            document_chunks = []
    else:
        # Initialize empty index
        faiss_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
        document_chunks = []

# Save index
def save_index():
    faiss.write_index(faiss_index, f"{DB_FOLDER}/index.faiss")  # Change index to faiss_index
    with open(f"{DB_FOLDER}/chunks.json", 'w') as f:
        json.dump(document_chunks, f)
    print(f"Index saved with {len(document_chunks)} chunks")

def auto_tag_chunk(text):
    """Automatically assign role tags to chunks based on content analysis"""
    text_lower = text.lower()
    assigned_tags = []
    
    # Contract Amount Detection
    if any(phrase in text for phrase in ["Approved Budget Cost", "Total Budget", "Contract Amount"]):
        assigned_tags.append("CONTRACT_AMOUNT")
    
    # Technical Specs Detection
    if "abc:" in text_lower and any(item in text_lower for item in ["unit", "pcs", "piece", "specifications"]):
        assigned_tags.append("TECHNICAL_SPECS")
    
    # Timeline Detection
    if any(phrase in text_lower for phrase in ["closing", "deadline", "submission", "on or before", "date", "time"]):
        assigned_tags.append("TIMELINE")
    
    # Compliance Requirements Detection
    if any(phrase in text_lower for phrase in ["republic act", "compliance", "criteria", "eligibility", "requirements", "regulation"]):
        assigned_tags.append("COMPLIANCE_REQUIREMENTS")
    
    # Bidder Info Detection
    if any(phrase in text_lower for phrase in ["bidder", "supplier", "quotation", "philgeps", "mayor's permit", "documents"]):
        assigned_tags.append("BIDDER_INFO")
    
    # Contact Info Detection
    if any(phrase in text_lower for phrase in ["phone", "telefax", "email", "@", "contact", "inquiries", "office"]):
        assigned_tags.append("CONTACT_INFO")
    
    # Reference Info Detection
    if any(phrase in text_lower for phrase in ["reference", "rfq", "pr-", "procurement", "number", "id"]):
        assigned_tags.append("REFERENCE_INFO")
    
    return assigned_tags if assigned_tags else ["GENERAL"]

def determine_chunk_priority(text, tags):
    """Determine chunk priority based on content"""
    text_lower = text.lower()
    
    # High priority indicators
    if any(phrase in text_lower for phrase in ["approved budget cost", "closing", "republic act", "request for quotation"]):
        return "high"
    
    # Low priority indicators  
    if any(phrase in text_lower for phrase in ["contact", "phone", "email only"]):
        return "low"
    
    return "medium"

def determine_content_type(text, tags):
    """Determine content type based on text analysis"""
    text_lower = text.lower()
    
    if any(phrase in text_lower for phrase in ["request for quotation", "approved budget cost", "project"]):
        return "summary"
    
    if any(phrase in text_lower for phrase in ["republic act", "regulation", "criteria"]):
        return "regulatory"
    
    if any(phrase in text_lower for phrase in ["submission", "quotation", "documents"]):
        return "procedural"
    
    return "detailed"

def calculate_role_relevance(text, tags):
    """Calculate role relevance scores based on content"""
    text_lower = text.lower()
    relevance = {
        "auditor": 0.5,
        "procurement_officer": 0.5, 
        "bidder": 0.5,
        "policy_maker": 0.5
    }
    
    # Auditor relevance
    if "COMPLIANCE_REQUIREMENTS" in tags or "CONTRACT_AMOUNT" in tags:
        relevance["auditor"] += 0.4
    
    # Procurement Officer relevance
    if "TIMELINE" in tags or "BIDDER_INFO" in tags:
        relevance["procurement_officer"] += 0.4
    
    # Bidder relevance
    if "TECHNICAL_SPECS" in tags or "BIDDER_INFO" in tags:
        relevance["bidder"] += 0.4
    
    # Policy Maker relevance
    if "COMPLIANCE_REQUIREMENTS" in tags or "CONTRACT_AMOUNT" in tags:
        relevance["policy_maker"] += 0.4
    
    # Cap at 1.0
    for role in relevance:
        relevance[role] = min(1.0, relevance[role])
    
    return relevance

# Enhanced text chunking function with auto-tagging
def chunk_text(text, filename="", page_num=0):
    chunks = []
    i = 0
    while i < len(text):
        # Get chunk with overlap
        chunk_text = text[i:i + CHUNK_SIZE]
        if chunk_text:
            # Auto-generate tags and metadata
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
    global faiss_index, model
    
    chunks = extract_pdf_text(file_path)
    if not chunks:
        return {"success": False, "message": "No text extracted from document"}
    
    # Get embeddings for all chunks
    texts = [chunk["text"] for chunk in chunks]
    
    # Combine with existing texts if any
    all_existing_texts = [chunk["text"] for chunk in document_chunks]
    all_texts = all_existing_texts + texts
    
    # Refit the model on all texts (existing + new)
    model.fit(all_texts)
    
    # Get embeddings for new chunks only
    embeddings = model.encode(texts)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # If this is the first document or model dimension changed, rebuild index
    if not model.is_fitted or faiss_index.d != embeddings.shape[1]:
        print(f"Rebuilding index with dimension {embeddings.shape[1]}")
        faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        
        # Re-encode and add all existing chunks
        if document_chunks:
            all_embeddings = model.encode(all_existing_texts + texts)
            faiss.normalize_L2(all_embeddings)
            faiss_index.add(all_embeddings)
        else:
            faiss_index.add(embeddings)
    else:
        # Just add new embeddings
        faiss_index.add(embeddings)
    
    # Store chunk information
    for chunk in chunks:
        document_chunks.append(chunk)
    
    save_index()
    return {"success": True, "chunks_added": len(chunks)}

# Role-aware configurations - Updated with your specified roles
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

# role configurations with strict keyword mappings
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
    """Detect the primary intent/tag of the query based on keywords - Enhanced version"""
    query_lower = query.lower()
    
    # Score each tag based on keyword matches
    tag_scores = defaultdict(int)
    
    # Enhanced scoring system
    for tag, keywords in STRICT_TAG_KEYWORDS.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact phrase match (highest score)
            if keyword_lower in query_lower:
                # Check for word boundaries to avoid partial matches
                import re
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                if re.search(pattern, query_lower):
                    tag_scores[tag] += 5
                else:
                    # Partial match but still relevant
                    tag_scores[tag] += 2
            
            # Check for synonym/related word matches
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
    
    # Return the highest scoring tag or None if no clear match
    if tag_scores:
        best_tag = max(tag_scores, key=tag_scores.get)
        best_score = tag_scores[best_tag]
        
        print(f"DEBUG: Intent detection - Query: '{query}', Scores: {dict(tag_scores)}, Best: {best_tag}({best_score})")
        
        if best_score >= 2:  # Minimum threshold for strict filtering
            return best_tag, best_score
    
    print(f"DEBUG: Intent detection - No clear intent found for query: '{query}'")
    return None, 0

def apply_strict_tag_filtering(chunks, required_tag, fallback_allowed=True):
    """Apply strict tag filtering - only return chunks with the required tag"""
    if not required_tag:
        return chunks
    
    filtered_chunks = []
    
    for chunk in chunks:
        chunk_tags = chunk.get('metadata', {}).get('role_tags', [])
        if required_tag in chunk_tags:
            filtered_chunks.append(chunk)
    
    # If no chunks found and fallback is allowed, return original chunks
    if not filtered_chunks and fallback_allowed:
        return chunks
    
    return filtered_chunks

def prioritize_chunks(chunks, role):
    """Prioritize chunks based on content type and role preferences - Legacy function"""
    # Just call the enhanced version
    return prioritize_chunks_enhanced(chunks, role)

def enhanced_chunk_filtering(query, role, all_chunks, top_k=5):
    """Enhanced filtering with better role-specific prioritization"""
    
    print(f"DEBUG: enhanced_chunk_filtering called with role: '{role}', query: '{query}'")
    role_logger.info(f"FILTERING_START: Role='{role}', Query='{query}', Total_chunks={len(all_chunks)}")
    
    # Step 1: Detect query intent
    required_tag, confidence = detect_query_intent(query)
    print(f"DEBUG: Detected intent: {required_tag} with confidence: {confidence}")
    
    # Step 2: Apply semantic similarity filtering first
    candidate_chunks = []
    if model.is_fitted and len(all_chunks) > 0:
        query_embedding = model.encode([query])
        faiss.normalize_L2(query_embedding)
        D, I = faiss_index.search(query_embedding, min(len(all_chunks), top_k * 3))
        
        # Get initial candidate chunks
        for idx in I[0]:
            if idx < len(all_chunks):
                candidate_chunks.append(all_chunks[idx])
    else:
        # If no embeddings available, use all chunks
        candidate_chunks = all_chunks[:top_k * 3]
    
    print(f"DEBUG: Initial candidates from semantic search: {len(candidate_chunks)}")
    
    # Step 3: Apply strict tag filtering if intent is clear
    used_fallback = False
    if required_tag and confidence >= 2:
        print(f"DEBUG: Applying strict tag filtering for tag: {required_tag}")
        filtered_chunks = apply_strict_tag_filtering(candidate_chunks, required_tag, fallback_allowed=True)
        print(f"DEBUG: After tag filtering: {len(filtered_chunks)} chunks")
        
        if len(filtered_chunks) < len(candidate_chunks):
            # Tag filtering was applied
            candidate_chunks = filtered_chunks
            print(f"DEBUG: Tag filtering reduced chunks to: {len(candidate_chunks)}")
        if not filtered_chunks:
            used_fallback = True
            candidate_chunks = apply_strict_tag_filtering(candidate_chunks, required_tag, fallback_allowed=True)
            print(f"DEBUG: Used fallback, chunks: {len(candidate_chunks)}")
    
    # Step 4: Apply role-specific prioritization - This is the KEY improvement
    print(f"DEBUG: Applying role-specific prioritization for role: {role}")
    prioritized_chunks = prioritize_chunks_enhanced(candidate_chunks, role)
    
    # Step 5: Take top_k chunks
    final_chunks = prioritized_chunks[:top_k]
    
    # Step 6: Add role-specific context framing
    role_context = ROLE_CONTEXTS.get(role, {})
    
    # Log the filtering results
    role_logger.info(f"FILTERING_RESULT: Role='{role}', Query='{query}', Intent='{required_tag}', Confidence={confidence}, Final_chunks={len(final_chunks)}")
    
    return {
        'chunks': final_chunks,
        'detected_intent': required_tag,
        'intent_confidence': confidence,
        'used_fallback': used_fallback,
        'role_framing': role_context.get('framing', ''),
        'role_emphasis': role_context.get('emphasis', ''),
        'total_candidates': len(candidate_chunks),
        'filtering_applied': required_tag is not None
    }

def prioritize_chunks_enhanced(chunks, role):
    """Enhanced chunk prioritization with better role-specific scoring"""
    if not chunks:
        return chunks
    
    print(f"DEBUG: prioritize_chunks_enhanced called with role: '{role}', chunks: {len(chunks)}")
    
    # Get role context for priority tags
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
        
        # Content type scoring (prefer summary over detailed for most roles)
        content_type = metadata.get('content_type', 'detailed')
        if role == 'policy_maker' and content_type in ['summary', 'regulatory']:
            score += 2
        elif role == 'auditor' and content_type in ['regulatory', 'procedural']:
            score += 2
        elif content_type == 'summary':
            score += 1
        
        # Role-specific tag preferences - MAJOR ENHANCEMENT
        chunk_tags = metadata.get('role_tags', [])
        tag_bonus = 0
        for tag in chunk_tags:
            if tag in priority_tags:
                tag_bonus += 2  # Significant bonus for matching priority tags
        score += tag_bonus
        
        # Role relevance score - CRITICAL FIX: Use the ACTUAL role, not 'general'
        role_relevance = metadata.get('role_relevance', {}).get(role, 0.5)
        relevance_bonus = role_relevance * 3  # Scale up the relevance impact
        score += relevance_bonus
        
        print(f"DEBUG: Chunk scoring - Tags: {chunk_tags}, Priority_tags: {priority_tags}, Tag_bonus: {tag_bonus}, Role_relevance: {role_relevance}, Total_score: {score}")
        
        return score
    
    # Sort by score (highest first) and return
    scored_chunks = sorted(chunks, key=chunk_score, reverse=True)
    
    # Log the top scores for debugging
    for i, chunk in enumerate(scored_chunks[:3]):
        score = chunk_score(chunk)
        tags = chunk.get('metadata', {}).get('role_tags', [])
        relevance = chunk.get('metadata', {}).get('role_relevance', {}).get(role, 0.5)
        print(f"DEBUG: Top chunk {i+1}: Score={score:.2f}, Tags={tags}, Relevance={relevance:.2f}")
    
    return scored_chunks

def create_fallback_response(query, role):
    """Create a structured fallback response when no relevant chunks are found"""
    role_context = ROLE_CONTEXTS.get(role, {})
    
    fallback_message = f"""I couldn't find specific information to answer your query: "{query}".

{role_context.get('framing', 'From your perspective,')} I recommend:

1. Checking if your question relates to information that might be in other procurement documents
2. Contacting the BAC Secretariat at (062) 991-1771 loc 1003 or bac@wmsu.edu.ph for clarification
3. Reviewing the complete procurement documents for comprehensive information

Please rephrase your question or ask about specific aspects like budget, timeline, specifications, or compliance requirements."""
    
    return fallback_message

# Add logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('role_awareness.log'),
        logging.StreamHandler()
    ]
)

# Create specialized loggers
role_logger = logging.getLogger('role_awareness')

def log_query_result(query, role, detected_tag, confidence, chunks_used, role_relevance_scores, tags_used):
    """Enhanced logging of query processing results with tag and relevance details"""
    
    # Create a clear separator for each new query
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

# Enhanced query function with role-aware filtering
def query_ollama_with_role(query, role='general', top_k=5):
    # Get query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS
    D, I = faiss_index.search(query_embedding, top_k * 2)  # Get more results to filter
    
    if len(I[0]) == 0:
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Filter and score contexts based on role
    contexts = []
    role_tags = ROLE_TAGS.get(role, [])
    
    for idx in I[0]:
        if idx < len(document_chunks):
            chunk = document_chunks[idx]
            
            # Calculate role relevance score
            relevance_score = 1.0  # Default score
            if 'role_tags' in chunk.get('metadata', {}):
                chunk_tags = chunk['metadata']['role_tags']
                # Boost score if chunk has relevant role tags
                tag_match = len(set(chunk_tags) & set(role_tags))
                if tag_match > 0:
                    relevance_score = 1.0 + (tag_match * 0.3)  # Boost by 30% per matching tag
            
            if 'role_relevance' in chunk.get('metadata', {}):
                role_relevance = chunk['metadata']['role_relevance'].get(role, 0.5)
                relevance_score *= role_relevance
            
            contexts.append({
                'text': chunk['text'],
                'score': relevance_score,
                'tags': chunk.get('metadata', {}).get('role_tags', []),
                'source': chunk.get('metadata', {}).get('source', 'unknown')
            })
    
    # Sort by relevance score and take top_k
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

# Enhanced query function with strict filtering
def query_ollama_with_strict_filtering(query, role='general', top_k=5):
    """Enhanced query function with strict tag filtering and role-specific processing"""
    
    print(f"DEBUG: query_ollama_with_strict_filtering called with role: '{role}', query: '{query}'")
    role_logger.info(f"QUERY_START: Role='{role}', Query='{query}', Available_chunks={len(document_chunks)}")
    
    if len(document_chunks) == 0:
        log_query_result(query, role, None, 0, 0, [], [])
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Apply enhanced filtering
    filter_result = enhanced_chunk_filtering(query, role, document_chunks, top_k)
    
    chunks = filter_result['chunks']

    # Extract role relevance scores for the SPECIFIC role (not general)
    role_relevance_scores = []
    all_tags_used = set()
    
    for chunk in chunks:
        # Get role relevance score for the CURRENT SELECTED role
        relevance = chunk.get('metadata', {}).get('role_relevance', {}).get(role, 0.5)
        role_relevance_scores.append(relevance)
        
        # Collect all tags used in selected chunks
        chunk_tags = chunk.get('metadata', {}).get('role_tags', [])
        all_tags_used.update(chunk_tags)

    tags_used_list = list(all_tags_used)

    print(f"DEBUG: Role relevance scores for role '{role}': {role_relevance_scores}")
    print(f"DEBUG: Tags used: {tags_used_list}")
    print(f"DEBUG: Number of chunks selected: {len(chunks)}")

    # Check if we have relevant chunks
    if not chunks:
        log_query_result(
            query,
            role,
            filter_result.get('detected_intent'),
            filter_result.get('intent_confidence', 0),
            0,
            [],
            []
        )
        fallback_response = create_fallback_response(query, role)
        return {
            "response": fallback_response,
            "role": role,
            "contexts_used": 0,
            "filtering_info": filter_result,
            "is_fallback": True
        }
    
    # Build enhanced context with role-specific framing
    contexts_text = []
    for chunk in chunks:
        contexts_text.append(chunk['text'])
    
    context_text = "\n\n".join(contexts_text)
    
    # Build role-specific prompt with better context
    role_framing = filter_result.get('role_framing', '')
    role_emphasis = filter_result.get('role_emphasis', '')
    detected_intent = filter_result.get('detected_intent', '')
    
    # Enhanced role-specific prompting
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
    
    {f"The query appears to be about: {detected_intent}" if detected_intent else ""}
    
    Context information:
    {context_text}
    
    Query: {query}
    
    Provide a clear, role-appropriate answer based only on the provided context. If the information is not in the context, say that you don't know but suggest relevant next steps.
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
            # Log AFTER successful response generation with the CORRECT role
            log_query_result(
                query,
                role,  # Use the actual selected role
                filter_result.get('detected_intent'),
                filter_result.get('intent_confidence', 0),
                len(chunks),
                role_relevance_scores,  # These should now be for the correct role
                tags_used_list
            )
            
            # Also log detailed chunk analysis with correct role focus
            role_logger.info(f"CHUNK_ANALYSIS for query '{query}' with role '{role}':")
            for i, chunk in enumerate(chunks):
                chunk_tags = chunk.get('metadata', {}).get('role_tags', [])
                chunk_relevance = chunk.get('metadata', {}).get('role_relevance', {})
                role_specific_relevance = chunk_relevance.get(role, 0.5)
                role_logger.info(f"  Chunk {i+1}: TAGS={chunk_tags} | {role.upper()}_RELEVANCE={role_specific_relevance:.3f} | ALL_RELEVANCE={chunk_relevance}")
            
            # Log role-specific effectiveness
            avg_relevance = sum(role_relevance_scores) / len(role_relevance_scores) if role_relevance_scores else 0.5
            role_logger.info(f"ROLE_EFFECTIVENESS: Role='{role}', Avg_Relevance={avg_relevance:.3f}, Tags_Found={len(tags_used_list)}, Intent_Match={detected_intent}")
            
            response_data = {
                "response": response.json()["response"],
                "role": role,
                "confirmed_role": role,  # Add explicit confirmation
                "contexts_used": len(chunks),
                "relevant_tags": list(set([tag for chunk in chunks for tag in chunk.get('metadata', {}).get('role_tags', [])])),
                "sources": list(set([chunk.get('metadata', {}).get('source', 'unknown') for chunk in chunks])),
                "filtering_info": filter_result,
                "is_fallback": False,
                "role_effectiveness": {
                    "avg_relevance": avg_relevance,
                    "tags_found": len(tags_used_list),
                    "intent_detected": detected_intent
                }
            }
            
            return response_data
        else:
            role_logger.error(f"Ollama error: {response.text}")
            return {"error": f"Ollama error: {response.text}"}
    except Exception as e:
        role_logger.error(f"Error querying Ollama: {str(e)}")
        return {"error": f"Error querying Ollama: {str(e)}"}

# Flask routes
@app.route('/')
def serve_index():  # Renamed from 'index' to 'serve_index'
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload endpoint called!")  # Debug
    
    if 'file' not in request.files:
        print("No file part in request")  # Debug
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("No selected file")  # Debug
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        print(f"Invalid file type: {file.filename}")  # Debug
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    print(f"Processing file: {file.filename}")  # Debug
    
    # Save file temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)
    print(f"File saved to: {temp_path}")  # Debug
    
    # Process file
    result = add_document_to_index(temp_path)
    print(f"Processing result: {result}")  # Debug
    
    return jsonify(result)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400

    # Extract role more carefully with better validation
    role = data.get('role', 'general')
    
    # Debug logging to track role (terminal only)
    print(f"DEBUG: Received request data: {data}")
    print(f"DEBUG: Extracted role: '{role}' (type: {type(role)})")
    
    # Clean up role string and ensure it's valid
    if isinstance(role, str):
        role = role.strip().lower()
    
    # Map frontend values to backend values if needed
    role_mapping = {
        'general': 'general',
        'auditor': 'auditor',
        'procurement_officer': 'procurement_officer',
        'policy_maker': 'policy_maker',
        'bidder': 'bidder'
    }
    
    # Ensure role is valid
    if role in role_mapping:
        role = role_mapping[role]
    else:
        print(f"DEBUG: Invalid role '{role}', defaulting to 'general'")
        role_logger.warning(f"Invalid role '{role}', defaulting to 'general'")
        role = 'general'
    
    print(f"DEBUG: Final role being used: '{role}'")
    
    # Log role switching very prominently
    role_logger.info("🔄" + "="*50 + "🔄")
    role_logger.info(f"🎭 ROLE SWITCH: Now operating as '{role.upper()}'")
    role_logger.info(f"📝 Query: '{data['query']}'")
    role_logger.info("🔄" + "="*50 + "🔄")
    
    role_logger.info(f"RECEIVED_ROLE: '{role}' for query: '{data['query']}'")
    
    # Call the enhanced query function with explicit role confirmation
    result = query_ollama_with_strict_filtering(data['query'], role)
    
    # Add role confirmation to response for debugging
    clean_result = {
        "response": result.get("response", ""),
        "role": role,
        "confirmed_role": role,  # Explicit confirmation
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
    """Get available roles for the dropdown"""
    return jsonify({
        "roles": list(ROLE_TAGS.keys()),
        "role_descriptions": ROLE_PROMPTS
    })

@app.route('/analyze-chunks', methods=['GET'])
def analyze_chunks():
    """Analyze the current chunks for role-aware tags"""
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
    """Analyze tagging consistency across all documents"""
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
    
    # Convert sets to lists for JSON serialization
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