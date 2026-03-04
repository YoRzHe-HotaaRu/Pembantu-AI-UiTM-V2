"""
Retriever - Combines keyword and semantic search for hybrid retrieval
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RetrievedChunk:
    """Represents a retrieved chunk with relevance info"""
    id: str
    content: str
    doc_title: str
    category: str
    chunk_index: int
    total_chunks: int
    similarity_score: float
    keyword_score: float
    combined_score: float


class HybridRetriever:
    """
    Hybrid retriever that combines:
    1. Semantic search (vector similarity)
    2. Keyword matching (BM25-like)
    """
    
    def __init__(
        self,
        vector_store,
        embedding_engine,
        document_loader,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        keyword_boost: float = 0.2
    ):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: VectorStore instance
            embedding_engine: EmbeddingEngine instance
            document_loader: DocumentLoader instance
            semantic_weight: Weight for semantic search scores
            keyword_weight: Weight for keyword matching scores
            keyword_boost: Boost for exact keyword matches
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.document_loader = document_loader
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.keyword_boost = keyword_boost
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of results to return
            category_filter: Optional category to filter by
            
        Returns:
            List of RetrievedChunk objects
        """
        # Step 1: Semantic search
        semantic_results = self._semantic_search(query, top_k * 2, category_filter)
        
        # Step 2: Keyword search
        keyword_results = self._keyword_search(query, category_filter)
        
        # Step 3: Combine and rerank
        combined_results = self._combine_results(
            semantic_results,
            keyword_results,
            query,
            top_k
        )
        
        return combined_results
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        category_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        # Generate query embedding
        query_embedding = self.embedding_engine.embed_text(query)
        
        # Build filter
        filter_dict = None
        if category_filter:
            filter_dict = {'category': category_filter}
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        category_filter: Optional[str]
    ) -> Dict[str, float]:
        """
        Perform keyword-based search
        Returns dict mapping chunk_id to keyword score
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return {}
        
        # Search across documents
        keyword_scores = {}
        
        for doc in self.document_loader.documents:
            # Filter by category if specified
            if category_filter and doc.category != category_filter:
                continue
            
            # Score document
            doc_score = self._score_keywords_in_text(keywords, doc.content)
            
            # If document has good score, find specific chunks
            if doc_score > 0:
                # For now, assign score to all chunks of this doc
                # In a more sophisticated version, we'd search within the doc
                for i in range(10):  # Assume max 10 chunks
                    chunk_id = f"{doc.id}#{i}"
                    keyword_scores[chunk_id] = doc_score
        
        return keyword_scores
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common stopwords in Malay and English
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must',
            'yang', 'dan', 'atau', 'untuk', 'dari', 'dengan', 'pada',
            'ke', 'di', 'ini', 'itu', 'saya', 'anda', 'dia', 'kita',
            'mereka', 'apa', 'siapa', 'bagaimana', 'bila', 'kenapa'
        }
        
        # Split and clean
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _score_keywords_in_text(self, keywords: List[str], text: str) -> float:
        """Score how well keywords match in text"""
        text_lower = text.lower()
        
        score = 0.0
        for keyword in keywords:
            # Count occurrences
            count = text_lower.count(keyword)
            
            # Boost for exact matches
            if keyword in text_lower:
                score += count * 1.0
                
                # Extra boost if keyword appears in first 100 chars
                if keyword in text_lower[:100]:
                    score += self.keyword_boost
        
        # Normalize by keyword count
        if keywords:
            score /= len(keywords)
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_scores: Dict[str, float],
        query: str,
        top_k: int
    ) -> List[RetrievedChunk]:
        """Combine and rerank semantic and keyword results"""
        # Build combined score dictionary
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result['id']
            semantic_score = result['similarity']
            keyword_score = keyword_scores.get(chunk_id, 0.0)
            
            # Apply exact match boost
            if self._has_exact_match(query, result['content']):
                keyword_score = max(keyword_score, 0.5)
            
            combined_score = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )
            
            combined[chunk_id] = {
                'semantic_score': semantic_score,
                'keyword_score': keyword_score,
                'combined_score': combined_score,
                'content': result['content'],
                'metadata': result['metadata']
            }
        
        # Add keyword-only results that weren't in semantic results
        for chunk_id, keyword_score in keyword_scores.items():
            if chunk_id not in combined and keyword_score > 0.3:
                # Fetch from vector store
                chunk_data = self.vector_store.get_chunk_by_id(chunk_id)
                if chunk_data:
                    combined_score = self.keyword_weight * keyword_score
                    
                    combined[chunk_id] = {
                        'semantic_score': 0.0,
                        'keyword_score': keyword_score,
                        'combined_score': combined_score,
                        'content': chunk_data['content'],
                        'metadata': chunk_data['metadata']
                    }
        
        # Sort by combined score
        sorted_results = sorted(
            combined.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )[:top_k]
        
        # Build RetrievedChunk objects
        retrieved_chunks = []
        for chunk_id, data in sorted_results:
            meta = data['metadata']
            
            retrieved_chunks.append(RetrievedChunk(
                id=chunk_id,
                content=data['content'],
                doc_title=meta.get('doc_title', 'Unknown'),
                category=meta.get('category', 'unknown'),
                chunk_index=meta.get('chunk_index', 0),
                total_chunks=meta.get('total_chunks', 1),
                similarity_score=data['semantic_score'],
                keyword_score=data['keyword_score'],
                combined_score=data['combined_score']
            ))
        
        return retrieved_chunks
    
    def _has_exact_match(self, query: str, content: str) -> bool:
        """Check if query has exact or near-exact match in content"""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Check for phrase match
        if query_lower in content_lower:
            return True
        
        # Check for high word overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        if len(query_words) > 0:
            overlap = len(query_words & content_words) / len(query_words)
            return overlap > 0.7
        
        return False
    
    def format_context(
        self,
        chunks: List[RetrievedChunk],
        max_tokens: int = 2000
    ) -> str:
        """
        Format retrieved chunks into context for LLM
        
        Args:
            chunks: Retrieved chunks
            max_tokens: Maximum tokens (approximate)
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            # Format chunk
            formatted = f"""
[Sumber: {chunk.doc_title}]
{chunk.content}
---
"""
            
            # Rough token estimate (chars / 4)
            chunk_tokens = len(formatted) // 4
            
            if current_length + chunk_tokens > max_tokens:
                break
            
            context_parts.append(formatted)
            current_length += chunk_tokens
        
        return "\n".join(context_parts)
