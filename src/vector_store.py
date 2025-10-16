"""
Stage 7: Vector Database Storage

This module implements vector database storage with support for Qdrant and Weaviate.
It stores embeddings with rich metadata for effective retrieval.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import uuid
from datetime import datetime

# Vector database clients
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, CollectionStatus, PointStruct,
        Filter, FieldCondition, MatchValue, SearchParams
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import weaviate
    from weaviate.classes.config import Configure
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from .models import (
    EmbeddingResult, VectorStoreMetadata, ConfigParameters,
    DocumentMetadata, ContentType
)


logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector database implementations."""
    
    def __init__(self, config: ConfigParameters):
        """Initialize vector store with configuration."""
        self.config = config
        self.collection_name = config.collection_name
        self.distance_metric = config.distance_metric
        self.dimension = config.embedding_dimension
    
    @abstractmethod
    async def initialize_collection(self) -> bool:
        """Initialize/create collection in vector database."""
        pass
    
    @abstractmethod
    async def store_embeddings(
        self, 
        embedding_results: List[EmbeddingResult],
        document_metadata: DocumentMetadata
    ) -> List[str]:
        """Store embeddings with metadata. Returns list of vector IDs."""
        pass
    
    @abstractmethod
    async def search_similar(
        self, 
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete_by_document(self, document_id: str) -> bool:
        """Delete all vectors for a specific document."""
        pass
    
    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector database implementation."""
    
    def __init__(self, config: ConfigParameters):
        """Initialize Qdrant vector store."""
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not available. Install with: pip install qdrant-client")
        
        super().__init__(config)
        
        # Initialize Qdrant client
        qdrant_config = config.custom_fields.get('qdrant', {})
        self.host = qdrant_config.get('host', 'localhost')
        self.port = qdrant_config.get('port', 6333)
        self.api_key = qdrant_config.get('api_key')
        
        self.client = AsyncQdrantClient(
            host=self.host,
            port=self.port,
            api_key=self.api_key
        )
    
    async def initialize_collection(self) -> bool:
        """Initialize Qdrant collection."""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name in existing_collections:
                logger.info(f"Qdrant collection '{self.collection_name}' already exists")
                return True
            
            # Create collection
            distance_map = {
                'cosine': Distance.COSINE,
                'euclidean': Distance.EUCLID,
                'dot': Distance.DOT
            }
            
            distance = distance_map.get(self.distance_metric.lower(), Distance.COSINE)
            
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=distance
                )
            )
            
            logger.info(f"Created Qdrant collection '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant collection: {e}")
            return False
    
    async def store_embeddings(
        self, 
        embedding_results: List[EmbeddingResult],
        document_metadata: DocumentMetadata
    ) -> List[str]:
        """Store embeddings in Qdrant."""
        try:
            points = []
            vector_ids = []
            
            for result in embedding_results:
                # Store text embedding
                text_vector_id = str(uuid.uuid4())
                text_metadata = self._create_vector_metadata(
                    result, document_metadata, "text", text_vector_id
                )
                
                text_point = PointStruct(
                    id=text_vector_id,
                    vector=result.text_embedding.embedding,
                    payload=text_metadata.dict()
                )
                points.append(text_point)
                vector_ids.append(text_vector_id)
                
                # Store table embedding if available
                if result.table_embedding:
                    table_vector_id = str(uuid.uuid4())
                    table_metadata = self._create_vector_metadata(
                        result, document_metadata, "table", table_vector_id
                    )
                    
                    table_point = PointStruct(
                        id=table_vector_id,
                        vector=result.table_embedding.embedding,
                        payload=table_metadata.dict()
                    )
                    points.append(table_point)
                    vector_ids.append(table_vector_id)
            
            # Batch insert points
            if points:
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                logger.debug(f"Stored {len(points)} vectors in Qdrant")
            
            return vector_ids
            
        except Exception as e:
            logger.error(f"Error storing embeddings in Qdrant: {e}")
            return []
    
    async def search_similar(
        self, 
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant."""
        try:
            # Build filter conditions
            filter_conditions = None
            if filters:
                filter_conditions = self._build_qdrant_filter(filters)
            
            # Perform search
            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter_conditions,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for point in search_result:
                result = {
                    'id': point.id,
                    'score': point.score,
                    'metadata': point.payload
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            return []
    
    async def delete_by_document(self, document_id: str) -> bool:
        """Delete all vectors for a specific document from Qdrant."""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"Deleted vectors for document {document_id} from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document vectors from Qdrant: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get Qdrant collection statistics."""
        try:
            collection_info = await self.client.get_collection(self.collection_name)
            
            stats = {
                'total_vectors': collection_info.points_count,
                'collection_status': collection_info.status,
                'vector_dimension': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance,
                'indexed_vectors': collection_info.indexed_vectors_count,
                'database_type': 'qdrant'
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting Qdrant collection stats: {e}")
            return {'error': str(e)}
    
    def _create_vector_metadata(
        self, 
        result: EmbeddingResult,
        document_metadata: DocumentMetadata,
        embedding_type: str,
        vector_id: str
    ) -> VectorStoreMetadata:
        """Create metadata object for vector storage."""
        chunk = result.chunk
        
        return VectorStoreMetadata(
            vector_id=vector_id,
            document_id=document_metadata.document_id,
            chunk_id=chunk.metadata.chunk_id,
            text=chunk.text,
            embedding_type=embedding_type,
            content_type=chunk.metadata.primary_content_type.value,
            is_table=chunk.metadata.is_table,
            page_numbers=chunk.metadata.page_numbers,
            chunk_position=chunk.metadata.chunk_position,
            section_ids=chunk.metadata.section_ids,
            section_titles=chunk.metadata.section_titles,
            section_hierarchy=chunk.metadata.section_hierarchy,
            ocr_source=chunk.metadata.primary_ocr_source.value,
            confidence_score=chunk.metadata.avg_confidence,
            token_count=chunk.metadata.token_count,
            word_count=chunk.metadata.word_count,
            char_count=chunk.metadata.char_count,
            doc_title=document_metadata.title,
            doc_type=document_metadata.doc_type,
            department=document_metadata.department,
            fiscal_year=document_metadata.fiscal_year,
            table_count=chunk.metadata.table_count,
            table_ids=chunk.metadata.table_ids,
            structured_data=self._serialize_structured_data(chunk.structured_data),
            created_at=chunk.metadata.created_at
        )
    
    def _serialize_structured_data(self, structured_data: List[Any]) -> Optional[Dict[str, Any]]:
        """Serialize structured table data for storage."""
        if not structured_data:
            return None
        
        try:
            # Convert first table to dictionary
            table = structured_data[0]
            if hasattr(table, 'dict'):
                return table.dict()
            else:
                return {
                    'headers': getattr(table, 'headers', []),
                    'rows': getattr(table, 'rows', [])[:5],  # Limit rows for storage
                    'caption': getattr(table, 'caption', None)
                }
        except Exception as e:
            logger.warning(f"Error serializing structured data: {e}")
            return None
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from filter dictionary."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values - use should (OR)
                or_conditions = [
                    FieldCondition(key=key, match=MatchValue(value=v))
                    for v in value
                ]
                conditions.extend(or_conditions)
            else:
                # Single value
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        
        return Filter(must=conditions) if conditions else None


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector database implementation."""
    
    def __init__(self, config: ConfigParameters):
        """Initialize Weaviate vector store."""
        if not WEAVIATE_AVAILABLE:
            raise ImportError("weaviate-client not available. Install with: pip install weaviate-client")
        
        super().__init__(config)
        
        # Initialize Weaviate client
        weaviate_config = config.custom_fields.get('weaviate', {})
        self.url = weaviate_config.get('url', 'http://localhost:8080')
        self.api_key = weaviate_config.get('api_key')
        
        # Configure client
        client_config = weaviate.Config(
            additional_headers={'X-OpenAI-Api-Key': weaviate_config.get('openai_key')} 
            if weaviate_config.get('openai_key') else None
        )
        
        self.client = weaviate.Client(self.url, auth_client_secret=None)
    
    async def initialize_collection(self) -> bool:
        """Initialize Weaviate collection (class)."""
        try:
            # Check if class exists
            schema = self.client.schema.get()
            existing_classes = [cls['class'] for cls in schema.get('classes', [])]
            
            if self.collection_name in existing_classes:
                logger.info(f"Weaviate class '{self.collection_name}' already exists")
                return True
            
            # Create class schema
            class_schema = {
                "class": self.collection_name,
                "description": "Document chunks with embeddings for RAG",
                "vectorizer": "none",  # We provide our own vectors
                "moduleConfig": {
                    "generative-openai": {
                        "model": "gpt-3.5-turbo"
                    }
                },
                "properties": [
                    {"name": "document_id", "dataType": ["text"]},
                    {"name": "chunk_id", "dataType": ["text"]},
                    {"name": "text", "dataType": ["text"]},
                    {"name": "embedding_type", "dataType": ["text"]},
                    {"name": "content_type", "dataType": ["text"]},
                    {"name": "is_table", "dataType": ["boolean"]},
                    {"name": "page_numbers", "dataType": ["int"]},
                    {"name": "chunk_position", "dataType": ["int"]},
                    {"name": "section_ids", "dataType": ["text"]},
                    {"name": "section_titles", "dataType": ["text"]},
                    {"name": "ocr_source", "dataType": ["text"]},
                    {"name": "confidence_score", "dataType": ["number"]},
                    {"name": "token_count", "dataType": ["int"]},
                    {"name": "word_count", "dataType": ["int"]},
                    {"name": "doc_title", "dataType": ["text"]},
                    {"name": "doc_type", "dataType": ["text"]},
                    {"name": "department", "dataType": ["text"]},
                    {"name": "fiscal_year", "dataType": ["int"]},
                    {"name": "created_at", "dataType": ["date"]}
                ]
            }
            
            self.client.schema.create_class(class_schema)
            logger.info(f"Created Weaviate class '{self.collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Weaviate class: {e}")
            return False
    
    async def store_embeddings(
        self, 
        embedding_results: List[EmbeddingResult],
        document_metadata: DocumentMetadata
    ) -> List[str]:
        """Store embeddings in Weaviate."""
        try:
            vector_ids = []
            
            for result in embedding_results:
                # Store text embedding
                text_vector_id = str(uuid.uuid4())
                text_metadata = self._create_vector_metadata(
                    result, document_metadata, "text", text_vector_id
                )
                
                # Convert metadata to Weaviate format
                text_properties = self._metadata_to_weaviate_properties(text_metadata)
                
                self.client.data_object.create(
                    data_object=text_properties,
                    class_name=self.collection_name,
                    uuid=text_vector_id,
                    vector=result.text_embedding.embedding
                )
                vector_ids.append(text_vector_id)
                
                # Store table embedding if available
                if result.table_embedding:
                    table_vector_id = str(uuid.uuid4())
                    table_metadata = self._create_vector_metadata(
                        result, document_metadata, "table", table_vector_id
                    )
                    
                    table_properties = self._metadata_to_weaviate_properties(table_metadata)
                    
                    self.client.data_object.create(
                        data_object=table_properties,
                        class_name=self.collection_name,
                        uuid=table_vector_id,
                        vector=result.table_embedding.embedding
                    )
                    vector_ids.append(table_vector_id)
            
            logger.debug(f"Stored {len(vector_ids)} vectors in Weaviate")
            return vector_ids
            
        except Exception as e:
            logger.error(f"Error storing embeddings in Weaviate: {e}")
            return []
    
    async def search_similar(
        self, 
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Weaviate."""
        try:
            # Build query
            query = self.client.query.get(self.collection_name, [
                "document_id", "chunk_id", "text", "embedding_type",
                "content_type", "section_titles", "doc_title"
            ]).with_near_vector({
                "vector": query_vector
            }).with_limit(limit)
            
            # Add filters if provided
            if filters:
                where_filter = self._build_weaviate_filter(filters)
                query = query.with_where(where_filter)
            
            # Execute query
            result = query.do()
            
            # Format results
            results = []
            objects = result.get('data', {}).get('Get', {}).get(self.collection_name, [])
            
            for obj in objects:
                result_obj = {
                    'id': obj.get('_additional', {}).get('id'),
                    'score': obj.get('_additional', {}).get('distance', 0),
                    'metadata': obj
                }
                results.append(result_obj)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in Weaviate: {e}")
            return []
    
    async def delete_by_document(self, document_id: str) -> bool:
        """Delete all vectors for a specific document from Weaviate."""
        try:
            where_filter = {
                "path": ["document_id"],
                "operator": "Equal",
                "valueText": document_id
            }
            
            self.client.batch.delete_objects(
                class_name=self.collection_name,
                where=where_filter
            )
            
            logger.info(f"Deleted vectors for document {document_id} from Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document vectors from Weaviate: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get Weaviate collection statistics."""
        try:
            # Get class info
            schema = self.client.schema.get(self.collection_name)
            
            # Get object count
            result = self.client.query.aggregate(self.collection_name).with_meta_count().do()
            count = result.get('data', {}).get('Aggregate', {}).get(self.collection_name, [{}])[0].get('meta', {}).get('count', 0)
            
            stats = {
                'total_vectors': count,
                'collection_status': 'active',
                'vector_dimension': self.dimension,
                'database_type': 'weaviate',
                'vectorizer': schema.get('vectorizer', 'none')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting Weaviate collection stats: {e}")
            return {'error': str(e)}
    
    def _metadata_to_weaviate_properties(self, metadata: VectorStoreMetadata) -> Dict[str, Any]:
        """Convert metadata to Weaviate properties format."""
        return {
            "document_id": metadata.document_id,
            "chunk_id": metadata.chunk_id,
            "text": metadata.text,
            "embedding_type": metadata.embedding_type,
            "content_type": metadata.content_type,
            "is_table": metadata.is_table,
            "page_numbers": metadata.page_numbers,
            "chunk_position": metadata.chunk_position,
            "section_ids": metadata.section_ids,
            "section_titles": metadata.section_titles,
            "ocr_source": metadata.ocr_source,
            "confidence_score": metadata.confidence_score,
            "token_count": metadata.token_count,
            "word_count": metadata.word_count,
            "doc_title": metadata.doc_title or "",
            "doc_type": metadata.doc_type or "",
            "department": metadata.department or "",
            "fiscal_year": metadata.fiscal_year,
            "created_at": metadata.created_at.isoformat()
        }
    
    def _build_weaviate_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build Weaviate filter from filter dictionary."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values - create OR conditions
                or_conditions = []
                for v in value:
                    or_conditions.append({
                        "path": [key],
                        "operator": "Equal",
                        "valueText": str(v)
                    })
                conditions.append({"operator": "Or", "operands": or_conditions})
            else:
                # Single value
                conditions.append({
                    "path": [key],
                    "operator": "Equal",
                    "valueText": str(value)
                })
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"operator": "And", "operands": conditions}
        else:
            return {}


class VectorStoreManager:
    """Manager for vector database operations."""
    
    def __init__(self, config: ConfigParameters):
        """Initialize vector store manager."""
        self.config = config
        
        # Initialize appropriate vector store
        if config.vector_store_type.lower() == 'qdrant':
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant client not available")
            self.store = QdrantVectorStore(config)
        elif config.vector_store_type.lower() == 'weaviate':
            if not WEAVIATE_AVAILABLE:
                raise ImportError("Weaviate client not available")
            self.store = WeaviateVectorStore(config)
        else:
            raise ValueError(f"Unsupported vector store type: {config.vector_store_type}")
        
        logger.info(f"Initialized {config.vector_store_type} vector store")
    
    async def initialize(self) -> bool:
        """Initialize the vector store."""
        return await self.store.initialize_collection()
    
    async def store_document_embeddings(
        self, 
        embedding_results: List[EmbeddingResult],
        document_metadata: DocumentMetadata
    ) -> List[str]:
        """Store all embeddings for a document."""
        return await self.store.store_embeddings(embedding_results, document_metadata)
    
    async def search(
        self, 
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        return await self.store.search_similar(query_vector, limit, filters)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all vectors for a document."""
        return await self.store.delete_by_document(document_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return await self.store.get_collection_stats()