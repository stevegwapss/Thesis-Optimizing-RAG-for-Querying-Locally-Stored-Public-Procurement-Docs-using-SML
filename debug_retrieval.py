"""
Debug script to test ChromaDB retrieval directly
Run this to see what's actually in the database and test similarity search
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# Configuration
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), 'db', 'chromadb')

print("=" * 80)
print("🔍 ChromaDB Retrieval Debugger")
print("=" * 80)

# Initialize embeddings
print("\n1️⃣ Initializing embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
print("✅ Embeddings initialized")

# Check if ChromaDB exists
if not os.path.exists(CHROMA_DB_PATH):
    print(f"\n❌ ChromaDB not found at: {CHROMA_DB_PATH}")
    print("Upload documents first!")
    exit(1)

# Load ChromaDB
print(f"\n2️⃣ Loading ChromaDB from: {CHROMA_DB_PATH}")
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings,
    collection_name="procurement_docs"
)
print("✅ ChromaDB loaded")

# Get database statistics
print("\n3️⃣ Database Statistics:")
try:
    collection = vectorstore._collection
    total_count = collection.count()
    print(f"   Total chunks: {total_count}")
    
    # Get all documents
    all_docs = vectorstore.get()
    
    if all_docs and 'metadatas' in all_docs:
        # Count chunks per file
        file_chunks = {}
        for metadata in all_docs['metadatas']:
            source_file = metadata.get('source_file', 'unknown')
            file_chunks[source_file] = file_chunks.get(source_file, 0) + 1
        
        print(f"   Unique files: {len(file_chunks)}")
        print("\n   Files in database:")
        for i, (filename, chunk_count) in enumerate(sorted(file_chunks.items()), 1):
            print(f"   {i}. {filename}: {chunk_count} chunks")
            
    # Show sample documents
    if all_docs and 'documents' in all_docs and len(all_docs['documents']) > 0:
        print(f"\n4️⃣ Sample Document Content (first 3 chunks):")
        for i in range(min(3, len(all_docs['documents']))):
            content = all_docs['documents'][i][:200]
            metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
            print(f"\n   Chunk {i+1}:")
            print(f"   Source: {metadata.get('source_file', 'unknown')}")
            print(f"   Page: {metadata.get('page', 'N/A')}")
            print(f"   Content: {content}...")
    
except Exception as e:
    print(f"   ❌ Error getting stats: {e}")

# Test similarity search
print("\n5️⃣ Testing Similarity Search:")
test_queries = [
    "procurement plan",
    "budget allocation",
    "specifications requirements",
]

for query in test_queries:
    print(f"\n   Query: '{query}'")
    try:
        results = vectorstore.similarity_search(query, k=3)
        print(f"   Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source_file', 'unknown')
            page = doc.metadata.get('page', 'N/A')
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"      {i}. {source} (page {page})")
            print(f"         Content: {content_preview}...")
    except Exception as e:
        print(f"      ❌ Search failed: {e}")

# Interactive query test
print("\n6️⃣ Interactive Query Test:")
print("   Enter a query to test (or 'quit' to exit):")
while True:
    user_query = input("\n   > ").strip()
    if user_query.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_query:
        continue
        
    try:
        print(f"   Searching for: '{user_query}'")
        results = vectorstore.similarity_search(user_query, k=5)
        
        if not results:
            print("   ❌ No results found")
            continue
            
        print(f"   ✅ Found {len(results)} results:\n")
        
        unique_files = set()
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source_file', 'unknown')
            page = doc.metadata.get('page', 'N/A')
            content_preview = doc.page_content[:200].replace('\n', ' ')
            unique_files.add(source)
            
            print(f"   Result {i}:")
            print(f"   Source: {source}")
            print(f"   Page: {page}")
            print(f"   Content: {content_preview}...")
            print()
        
        print(f"   📁 Unique files in results: {', '.join(unique_files)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 80)
print("Debug session ended")
print("=" * 80)
