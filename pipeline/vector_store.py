"""
Vector Store - Stage 7 of PDF Chunking Pipeline

ChromaDB implementation with:
- Collection management
- Metadata filtering  
- Hybrid search (vector + filters)
- Batch operations
- Rich metadata support
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import uuid
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available")

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    chunk_type: str  # 'text' or 'table'
    source_document: str
    page_number: int
    position: int
    confidence_score: float

@dataclass
class SearchResult:
    """Represents a search result."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    distance: float

class ChromaVectorStore:
    """ChromaDB vector store with advanced features."""
    
    def __init__(self, persist_directory: str = "./chroma_db", config: Dict = None):
        self.persist_directory = Path(persist_directory)
        self.config = config or {}
        self.client = None
        self.collections = {}
        
        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available")
            return
        
        try:
            # Configure ChromaDB settings
            settings = Settings(
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False,
                allow_reset=True
            )
            
            self.client = chromadb.Client(settings)
            logger.info(f"Initialized ChromaDB client with persistence at {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
    
    def is_available(self) -> bool:
        """Check if ChromaDB is available and initialized."""
        return CHROMADB_AVAILABLE and self.client is not None
    
    def create_collection(self, name: str, embedding_function=None, metadata: Dict = None) -> bool:
        """Create or get a collection."""
        if not self.is_available():
            logger.error("ChromaDB not available")
            return False
        
        try:
            # Default metadata
            collection_metadata = {
                "description": "Procurement documents collection",
                "created_at": time.time(),
                **(metadata or {})
            }
            
            # Create or get collection
            collection = self.client.get_or_create_collection(
                name=name,
                embedding_function=embedding_function,
                metadata=collection_metadata
            )
            
            self.collections[name] = collection
            logger.info(f"Created/retrieved collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            return False
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        if not self.is_available():
            return False
        
        try:
            self.client.delete_collection(name)
            if name in self.collections:
                del self.collections[name]
            logger.info(f"Deleted collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection {name}: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        if not self.is_available():
            return []
        
        try:
            collections = self.client.list_collections()
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[Chunk],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> bool:
        """Insert or update chunks in a collection."""
        if not self.is_available():
            return False
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return False
        
        if not chunks:
            logger.warning("No chunks to upsert")
            return True
        
        try:
            collection = self.collections[collection_name]
            
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
            
            # Handle embeddings
            if embeddings:
                embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]
            else:
                # Use embeddings from chunks if available
                embeddings_list = []
                for chunk in chunks:
                    if chunk.embedding is not None:
                        embeddings_list.append(chunk.embedding.tolist())
                    else:
                        embeddings_list.append(None)
                
                # Filter out None embeddings
                if any(emb is None for emb in embeddings_list):
                    logger.warning("Some chunks missing embeddings, will use ChromaDB default embedding function")
                    embeddings_list = None
            
            # Upsert to ChromaDB
            if embeddings_list:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings_list,
                    metadatas=metadatas
                )
            else:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"Upserted {len(chunks)} chunks to collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error upserting chunks to {collection_name}: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_embedding: Optional[np.ndarray] = None,
        filters: Optional[Dict] = None,
        top_k: int = 5,
        include_distances: bool = True
    ) -> List[SearchResult]:
        """Search collection with optional metadata filtering."""
        if not self.is_available():
            return []
        
        if collection_name not in self.collections:
            logger.error(f"Collection {collection_name} not found")
            return []
        
        try:
            collection = self.collections[collection_name]
            
            # Prepare query parameters
            query_params = {
                'n_results': top_k,
                'include': ['documents', 'metadatas', 'distances'] if include_distances else ['documents', 'metadatas']
            }
            
            # Add query (text or embedding)
            if query_embedding is not None:
                query_params['query_embeddings'] = [query_embedding.tolist()]
            elif query_text:
                query_params['query_texts'] = [query_text]
            else:
                logger.error("Either query_text or query_embedding must be provided")
                return []
            
            # Add filters
            if filters:
                query_params['where'] = filters
            
            # Execute query
            results = collection.query(**query_params)
            
            # Parse results
            return self._parse_search_results(results, include_distances)
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return []
    
    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        filters: Dict,
        top_k: int = 10,
        embedding_weight: float = 0.7,
        filter_weight: float = 0.3
    ) -> List[SearchResult]:
        """Hybrid search combining vector similarity and metadata filtering."""
        # For now, this is a simplified implementation
        # In the future, we could implement more sophisticated hybrid ranking
        
        # First, do vector search with filters
        vector_results = self.search(
            collection_name=collection_name,
            query_text=query,
            filters=filters,
            top_k=top_k
        )
        
        # Apply hybrid scoring (simplified)
        for result in vector_results:
            # Combine vector similarity with filter match score
            filter_score = self._calculate_filter_score(result.metadata, filters)
            result.score = (embedding_weight * (1 - result.distance)) + (filter_weight * filter_score)
        
        # Re-sort by hybrid score
        vector_results.sort(key=lambda x: x.score, reverse=True)
        
        return vector_results
    
    def get_by_id(self, collection_name: str, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        if not self.is_available():
            return None
        
        if collection_name not in self.collections:
            return None
        
        try:
            collection = self.collections[collection_name]
            results = collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                return SearchResult(
                    chunk_id=results['ids'][0],
                    text=results['documents'][0],
                    score=1.0,
                    metadata=results['metadatas'][0],
                    distance=0.0
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting chunk {chunk_id} from {collection_name}: {e}")
            return None
    
    def search_by_metadata(
        self,
        collection_name: str,
        filters: Dict,
        limit: int = 100
    ) -> List[SearchResult]:
        """Search by metadata filters only."""
        if not self.is_available():
            return []
        
        if collection_name not in self.collections:
            return []
        
        try:
            collection = self.collections[collection_name]
            results = collection.get(
                where=filters,
                limit=limit,
                include=['documents', 'metadatas']
            )
            
            search_results = []
            for i, chunk_id in enumerate(results['ids']):
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=results['documents'][i],
                    score=1.0,  # Metadata matches get perfect score
                    metadata=results['metadatas'][i],
                    distance=0.0
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching by metadata in {collection_name}: {e}")
            return []
    
    def delete_chunks(self, collection_name: str, chunk_ids: List[str]) -> bool:
        """Delete specific chunks from a collection."""
        if not self.is_available():
            return False
        
        if collection_name not in self.collections:
            return False
        
        try:
            collection = self.collections[collection_name]
            collection.delete(ids=chunk_ids)
            logger.info(f"Deleted {len(chunk_ids)} chunks from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chunks from {collection_name}: {e}")
            return False
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics about a collection."""
        if not self.is_available():
            return {}
        
        if collection_name not in self.collections:
            return {}
        
        try:
            collection = self.collections[collection_name]
            count = collection.count()
            
            # Get sample of metadata to understand structure
            sample_results = collection.get(limit=10, include=['metadatas'])
            
            metadata_keys = set()
            chunk_types = set()
            source_documents = set()
            
            for metadata in sample_results.get('metadatas', []):
                metadata_keys.update(metadata.keys())
                if 'chunk_type' in metadata:
                    chunk_types.add(metadata['chunk_type'])
                if 'source_document' in metadata:
                    source_documents.add(metadata['source_document'])
            
            return {
                'name': collection_name,
                'total_chunks': count,
                'metadata_keys': list(metadata_keys),
                'chunk_types': list(chunk_types),
                'source_documents': list(source_documents),
                'sample_size': len(sample_results.get('metadatas', []))
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for {collection_name}: {e}")
            return {}
    
    def _prepare_metadata(self, chunk: Chunk) -> Dict:
        """Prepare metadata for ChromaDB storage."""
        metadata = chunk.metadata.copy()
        
        # Add standard fields
        metadata.update({
            'chunk_type': chunk.chunk_type,
            'source_document': chunk.source_document,
            'page_number': chunk.page_number,
            'position': chunk.position,
            'confidence_score': chunk.confidence_score,
            'text_length': len(chunk.text)
        })
        
        # Ensure all values are JSON serializable
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metadata[key] = value.item()
            elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                metadata[key] = str(value)
        
        return metadata
    
    def _parse_search_results(self, results: Dict, include_distances: bool) -> List[SearchResult]:
        """Parse ChromaDB search results."""
        search_results = []
        
        ids = results.get('ids', [[]])[0]
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0] if include_distances else [0.0] * len(ids)
        
        for i, chunk_id in enumerate(ids):
            search_results.append(SearchResult(
                chunk_id=chunk_id,
                text=documents[i],
                score=1 - distances[i] if include_distances else 1.0,  # Convert distance to similarity
                metadata=metadatas[i],
                distance=distances[i] if include_distances else 0.0
            ))
        
        return search_results
    
    def _calculate_filter_score(self, metadata: Dict, filters: Dict) -> float:
        """Calculate how well metadata matches filters."""
        if not filters:
            return 1.0
        
        matches = 0
        total = len(filters)
        
        for key, expected_value in filters.items():
            if key in metadata:
                actual_value = metadata[key]
                if actual_value == expected_value:
                    matches += 1
                elif isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                    # For numeric values, use proximity
                    proximity = 1 - abs(expected_value - actual_value) / max(abs(expected_value), abs(actual_value), 1)
                    matches += max(0, proximity)
        
        return matches / total if total > 0 else 1.0
    
    def export_collection(self, collection_name: str, output_path: Path) -> bool:
        """Export collection to JSON file."""
        if not self.is_available():
            return False
        
        if collection_name not in self.collections:
            return False
        
        try:
            collection = self.collections[collection_name]
            
            # Get all data
            results = collection.get(include=['documents', 'metadatas', 'embeddings'])
            
            export_data = {
                'collection_name': collection_name,
                'export_timestamp': time.time(),
                'total_chunks': len(results['ids']),
                'chunks': []
            }
            
            for i, chunk_id in enumerate(results['ids']):
                chunk_data = {
                    'id': chunk_id,
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'embedding': results['embeddings'][i] if results.get('embeddings') else None
                }
                export_data['chunks'].append(chunk_data)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported collection {collection_name} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting collection {collection_name}: {e}")
            return False
    
    def import_collection(self, input_path: Path, collection_name: Optional[str] = None) -> bool:
        """Import collection from JSON file."""
        if not self.is_available():
            return False
        
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Use provided name or original name
            target_collection = collection_name or import_data['collection_name']
            
            # Create collection if it doesn't exist
            if target_collection not in self.collections:
                self.create_collection(target_collection)
            
            collection = self.collections[target_collection]
            
            # Prepare data for import
            chunks_data = import_data['chunks']
            
            ids = [chunk['id'] for chunk in chunks_data]
            documents = [chunk['document'] for chunk in chunks_data]
            metadatas = [chunk['metadata'] for chunk in chunks_data]
            embeddings = [chunk['embedding'] for chunk in chunks_data if chunk.get('embedding')]
            
            # Import data
            if embeddings and len(embeddings) == len(ids):
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
            else:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
            
            logger.info(f"Imported {len(chunks_data)} chunks to collection {target_collection}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing collection from {input_path}: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.client:
                # ChromaDB client cleanup (if any specific cleanup needed)
                pass
            logger.info("ChromaDB cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Utility functions for creating chunks
def create_text_chunk(
    text: str,
    source_document: str,
    page_number: int,
    position: int,
    metadata: Dict = None,
    confidence_score: float = 1.0
) -> Chunk:
    """Create a text chunk."""
    chunk_id = str(uuid.uuid4())
    
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        embedding=None,
        metadata=metadata or {},
        chunk_type='text',
        source_document=source_document,
        page_number=page_number,
        position=position,
        confidence_score=confidence_score
    )

def create_table_chunk(
    table_data: Dict,
    source_document: str,
    page_number: int,
    position: int,
    metadata: Dict = None,
    confidence_score: float = 1.0
) -> Chunk:
    """Create a table chunk."""
    chunk_id = str(uuid.uuid4())
    
    # Convert table to text representation
    text = _table_to_text(table_data)
    
    # Add table-specific metadata
    table_metadata = metadata or {}
    table_metadata.update({
        'table_id': table_data.get('id', 'unknown'),
        'row_count': len(table_data.get('rows', [])),
        'col_count': len(table_data.get('headers', [])),
        'table_type': table_data.get('metadata', {}).get('table_type', 'data')
    })
    
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        embedding=None,
        metadata=table_metadata,
        chunk_type='table',
        source_document=source_document,
        page_number=page_number,
        position=position,
        confidence_score=confidence_score
    )

def _table_to_text(table_data: Dict) -> str:
    """Convert table data to text representation."""
    headers = table_data.get('headers', [])
    rows = table_data.get('rows', [])
    
    text_parts = [f"Table with columns: {', '.join(headers)}"]
    
    for i, row in enumerate(rows[:3]):  # First 3 rows
        row_text = ', '.join([f"{k}: {v}" for k, v in row.items() if v])
        text_parts.append(f"Row {i+1}: {row_text}")
    
    if len(rows) > 3:
        text_parts.append(f"... and {len(rows) - 3} more rows")
    
    return '. '.join(text_parts)