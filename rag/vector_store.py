"""
Vector Store - Manages storage and retrieval of document embeddings
Uses ChromaDB for efficient similarity search
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class VectorStore:
    """
    Vector database for storing and searching document embeddings
    Uses ChromaDB as the backend
    """
    
    def __init__(
        self,
        collection_name: str = "uitm_knowledge",
        persist_directory: str = "rag_cache/chroma_db",
        embedding_dimension: int = 384  # Default for all-MiniLM-L6-v2
    ):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_dimension: Dimension of embedding vectors
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_dimension = embedding_dimension
        self.client = None
        self.collection = None
        self._initialized = False
    
    def _init_client(self):
        """Lazy initialization of ChromaDB client"""
        if self._initialized:
            return
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persist directory
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize client with persistence
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "UITM Knowledge Base"}
            )
            
            self._initialized = True
            print(f"Vector store initialized: {self.collection.count()} documents")
            
        except ImportError:
            raise ImportError(
                "chromadb not installed. "
                "Install with: pip install chromadb"
            )
    
    def add_chunks(
        self,
        chunks: List,
        embeddings: np.ndarray
    ):
        """
        Add chunks with their embeddings to the vector store
        
        Args:
            chunks: List of TextChunk objects
            embeddings: Array of embedding vectors
        """
        self._init_client()
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                'doc_id': chunk.doc_id,
                'doc_title': chunk.doc_title,
                'category': chunk.category,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks
            }
            for chunk in chunks
        ]
        
        # Convert embeddings to list for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]
            batch_embeds = embeddings_list[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_embeds
            )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return
            filter_dict: Optional filter criteria
            
        Returns:
            List of result dictionaries with chunk info and scores
        """
        self._init_client()
        
        # Convert to list for ChromaDB
        query_embedding_list = query_embedding.tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=top_k,
            where=filter_dict,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # Convert distance to similarity score (Chroma uses cosine distance)
                distance = results['distances'][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                formatted_results.append({
                    'id': doc_id,
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity
                })
        
        return formatted_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID"""
        self._init_client()
        
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas']
            )
            
            if result['ids']:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0]
                }
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id}: {e}")
        
        return None
    
    def delete_chunks_by_doc_id(self, doc_id: str):
        """Delete all chunks belonging to a document"""
        self._init_client()
        
        # Find all chunks with this doc_id
        results = self.collection.get(
            where={'doc_id': doc_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
    
    def clear(self):
        """Clear all data from the collection"""
        self._init_client()
        
        # Delete and recreate collection
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "UITM Knowledge Base"}
        )
        
        print("Vector store cleared")
    
    def count(self) -> int:
        """Get total number of chunks in the store"""
        self._init_client()
        return self.collection.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        self._init_client()
        
        count = self.collection.count()
        
        # Get unique documents
        all_meta = self.collection.get(include=['metadatas'])
        unique_docs = set()
        categories = {}
        
        if all_meta['metadatas']:
            for meta in all_meta['metadatas']:
                unique_docs.add(meta['doc_id'])
                cat = meta['category']
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_chunks': count,
            'total_documents': len(unique_docs),
            'categories': categories
        }
