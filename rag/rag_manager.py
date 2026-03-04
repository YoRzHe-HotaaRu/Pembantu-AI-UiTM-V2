"""
RAG Manager - Main orchestrator for the RAG system
Coordinates document loading, chunking, and retrieval
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .document_loader import DocumentLoader
from .simple_retriever import SimpleRetriever

# Optional imports for advanced features
try:
    from .chunker import TextChunker
    from .embeddings import EmbeddingEngine
    from .vector_store import VectorStore
    from .retriever import HybridRetriever
    ADVANCED_RAG_AVAILABLE = True
except ImportError:
    ADVANCED_RAG_AVAILABLE = False


class RAGManager:
    """
    Main manager for the RAG system
    Handles initialization, indexing, and query processing
    """
    
    def __init__(
        self,
        knowledge_base_path: str = "knowledge_base",
        cache_dir: str = "rag_cache",
        use_advanced: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Initialize RAG Manager
        
        Args:
            knowledge_base_path: Path to knowledge base folder
            cache_dir: Directory for caching
            use_advanced: Whether to use advanced embedding-based RAG (requires more resources)
            chunk_size: Size of text chunks (for advanced mode)
            chunk_overlap: Overlap between chunks (for advanced mode)
        """
        self.knowledge_base_path = knowledge_base_path
        self.cache_dir = cache_dir
        self.use_advanced = use_advanced and ADVANCED_RAG_AVAILABLE
        
        # Core components (always available)
        self.document_loader = DocumentLoader(knowledge_base_path)
        self.simple_retriever = None
        
        # Advanced components (optional)
        self.chunker = None
        self.embedding_engine = None
        self.vector_store = None
        self.hybrid_retriever = None
        
        self._initialized = False
        self._stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'categories': {}
        }
    
    def initialize(self, force_reindex: bool = False):
        """
        Initialize the RAG system
        
        Args:
            force_reindex: Whether to force reindexing
        """
        print("\n" + "="*60)
        print("Initializing RAG System")
        print("="*60)
        
        # Step 1: Load documents
        print("\n1. Loading documents...")
        documents = self.document_loader.load_all()
        
        if not documents:
            print("WARNING: No documents found in knowledge base!")
            print(f"Add documents to: {self.knowledge_base_path}")
            self._initialized = True
            return
        
        self._stats['total_documents'] = len(documents)
        print(f"   ✓ Loaded {len(documents)} documents")
        
        # Step 2: Build simple retriever (lightweight, always works)
        print("\n2. Building keyword index...")
        self.simple_retriever = SimpleRetriever(self.document_loader)
        self.simple_retriever.build_index()
        self._stats['total_chunks'] = len(self.simple_retriever.chunk_index)
        
        # Count categories
        for doc in documents:
            cat = doc.category
            self._stats['categories'][cat] = self._stats['categories'].get(cat, 0) + 1
        
        # Step 3: Advanced features (optional)
        if self.use_advanced:
            print("\n3. Initializing advanced features (embeddings)...")
            self._init_advanced_features(force_reindex)
        else:
            print("\n3. Advanced features disabled (using lightweight mode)")
            print("   Set use_advanced=True for semantic search (requires more RAM)")
        
        print("\n" + "="*60)
        print("RAG System Ready!")
        print(f"  Mode: {'Advanced (semantic + keyword)' if self.use_advanced else 'Lightweight (keyword only)'}")
        print(f"  Documents: {self._stats['total_documents']}")
        print(f"  Chunks: {self._stats['total_chunks']}")
        print(f"  Categories: {', '.join(self._stats['categories'].keys())}")
        print("="*60 + "\n")
        
        self._initialized = True
    
    def _init_advanced_features(self, force_reindex: bool = False):
        """Initialize embedding-based features (memory intensive)"""
        if not ADVANCED_RAG_AVAILABLE:
            print("   ⚠ Advanced dependencies not available")
            return
        
        try:
            # Initialize components
            self.chunker = TextChunker(
                chunk_size=500,
                chunk_overlap=100
            )
            self.embedding_engine = EmbeddingEngine(
                model_name="all-MiniLM-L6-v2",
                cache_dir=self.cache_dir
            )
            self.vector_store = VectorStore(
                persist_directory=f"{self.cache_dir}/chroma_db"
            )
            
            # Check if we need to index
            existing_count = self.vector_store.count()
            
            if existing_count > 0 and not force_reindex:
                print(f"   ✓ Vector store has {existing_count} chunks (skipped indexing)")
            else:
                # Chunk documents
                print("   Chunking documents...")
                chunks = self.chunker.chunk_documents(self.document_loader.documents)
                print(f"   ✓ Created {len(chunks)} chunks")
                
                # Generate embeddings in batches
                print("   Generating embeddings (this may take a while)...")
                batch_size = 32  # Process in small batches to save memory
                all_embeddings = []
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    batch_texts = [chunk.content for chunk in batch]
                    
                    print(f"     Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
                    embeddings = self.embedding_engine.embed_texts(batch_texts)
                    all_embeddings.append(embeddings)
                
                import numpy as np
                all_embeddings = np.vstack(all_embeddings)
                
                # Add to vector store
                print("   Storing in vector database...")
                self.vector_store.add_chunks(chunks, all_embeddings)
                self.embedding_engine.save_cache()
                
                print(f"   ✓ Indexed {len(chunks)} chunks")
            
            # Initialize hybrid retriever
            self.hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedding_engine=self.embedding_engine,
                document_loader=self.document_loader
            )
            print("   ✓ Advanced retriever ready")
            
        except Exception as e:
            print(f"   ⚠ Error initializing advanced features: {e}")
            print("   Falling back to lightweight mode")
            self.use_advanced = False
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        category_filter: Optional[str] = None,
        format_context: bool = True
    ) -> Dict[str, Any]:
        """
        Query the knowledge base
        
        Args:
            query_text: User query
            top_k: Number of results to retrieve
            category_filter: Optional category filter
            format_context: Whether to format results as context string
            
        Returns:
            Dictionary with results and context
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        # Use advanced retriever if available, otherwise simple
        if self.use_advanced and self.hybrid_retriever:
            return self._query_advanced(query_text, top_k, category_filter, format_context)
        else:
            return self._query_simple(query_text, top_k, category_filter, format_context)
    
    def _query_simple(
        self,
        query_text: str,
        top_k: int,
        category_filter: Optional[str],
        format_context: bool
    ) -> Dict[str, Any]:
        """Query using simple keyword-based retrieval"""
        chunks = self.simple_retriever.retrieve(
            query=query_text,
            top_k=top_k,
            category_filter=category_filter
        )
        
        # Format context
        context = ''
        if format_context and chunks:
            context = self.simple_retriever.format_context(chunks)
        
        # Extract sources
        sources = []
        for chunk in chunks:
            source = {
                'title': chunk.doc_title,
                'category': chunk.category,
                'relevance': round(chunk.score, 3)
            }
            if source not in sources:
                sources.append(source)
        
        return {
            'chunks': chunks,
            'context': context,
            'sources': sources
        }
    
    def _query_advanced(
        self,
        query_text: str,
        top_k: int,
        category_filter: Optional[str],
        format_context: bool
    ) -> Dict[str, Any]:
        """Query using advanced hybrid retrieval"""
        chunks = self.hybrid_retriever.retrieve(
            query=query_text,
            top_k=top_k,
            category_filter=category_filter
        )
        
        # Format context
        context = ''
        if format_context and chunks:
            context = self.hybrid_retriever.format_context(chunks)
        
        # Extract sources
        sources = []
        for chunk in chunks:
            source = {
                'title': chunk.doc_title,
                'category': chunk.category,
                'relevance': round(chunk.combined_score, 3)
            }
            if source not in sources:
                sources.append(source)
        
        return {
            'chunks': chunks,
            'context': context,
            'sources': sources
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        return self._stats
    
    def reload(self):
        """Reload and reindex the knowledge base"""
        print("\nReloading knowledge base...")
        self._initialized = False
        self._stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'categories': {}
        }
        self.initialize(force_reindex=True)
    
    def search_by_keyword(self, keyword: str) -> List[Any]:
        """Simple keyword search across documents"""
        return self.document_loader.search_by_keyword(keyword)
    
    def get_categories(self) -> List[str]:
        """Get list of available categories"""
        return list(self._stats['categories'].keys())
