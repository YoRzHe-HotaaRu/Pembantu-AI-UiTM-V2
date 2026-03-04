"""
Simple Retriever - Lightweight keyword-based retrieval
Does not require heavy embedding models
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class SimpleRetrievedChunk:
    """Represents a retrieved chunk with relevance info"""
    id: str
    content: str
    doc_title: str
    category: str
    score: float


class SimpleRetriever:
    """
    Lightweight retriever using TF-IDF-like keyword matching
    No heavy dependencies required
    """
    
    def __init__(self, document_loader):
        """
        Initialize simple retriever
        
        Args:
            document_loader: DocumentLoader instance with loaded documents
        """
        self.document_loader = document_loader
        self.chunk_index = []  # List of (chunk_id, content, metadata, word_freq)
        self._indexed = False
    
    def build_index(self):
        """Build search index from documents"""
        print("Building simple keyword index...")
        self.chunk_index = []
        
        for doc in self.document_loader.documents:
            # Split document into paragraphs/sections
            chunks = self._split_document(doc)
            
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{doc.id}#{i}"
                
                # Calculate word frequency
                word_freq = self._get_word_frequency(chunk_text)
                
                self.chunk_index.append({
                    'id': chunk_id,
                    'content': chunk_text,
                    'doc_id': doc.id,
                    'doc_title': doc.title,
                    'category': doc.category,
                    'word_freq': word_freq,
                    'total_words': sum(word_freq.values())
                })
        
        self._indexed = True
        print(f"  ✓ Indexed {len(self.chunk_index)} chunks")
    
    def _split_document(self, document, max_chunk_size: int = 1000):
        """Split document into chunks"""
        text = document.content
        
        # If short enough, return as single chunk
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_size = len(para)
            
            if current_size + para_size > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _get_word_frequency(self, text: str) -> Counter:
        """Get word frequency for a text"""
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stopwords to remove
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'her', 'way', 'many', 'oil', 'sit', 'set', 'run',
            'eat', 'far', 'sea', 'eye', 'ago', 'off', 'too', 'any', 'say', 'man',
            'try', 'ask', 'end', 'why', 'let', 'put', 'say', 'she', 'try', 'way',
            'own', 'say', 'too', 'old', 'tell', 'very', 'when', 'much', 'would',
            'there', 'their', 'what', 'said', 'each', 'which', 'will', 'about',
            'could', 'other', 'after', 'first', 'never', 'these', 'think', 'where',
            'being', 'every', 'great', 'might', 'shall', 'still', 'those', 'while',
            'this', 'that', 'have', 'from', 'they', 'know', 'want', 'been', 'good',
            'yang', 'dan', 'untuk', 'dengan', 'pada', 'adalah', 'dari', 'ini',
            'itu', 'kepada', 'oleh', 'bagi', 'serta', 'sebagai', 'dalam', 'atau'
        }
        
        # Filter out stopwords and short words
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        return Counter(filtered_words)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[str] = None
    ) -> List[SimpleRetrievedChunk]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of results to return
            category_filter: Optional category filter
            
        Returns:
            List of SimpleRetrievedChunk objects
        """
        if not self._indexed:
            self.build_index()
        
        # Get query word frequency
        query_freq = self._get_word_frequency(query)
        
        if not query_freq:
            return []
        
        # Score each chunk
        scores = []
        for chunk in self.chunk_index:
            # Filter by category if specified
            if category_filter and chunk['category'] != category_filter:
                continue
            
            # Calculate TF-IDF-like score
            score = self._calculate_score(query_freq, chunk)
            
            if score > 0:
                scores.append((chunk, score))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk, score in scores[:top_k]:
            results.append(SimpleRetrievedChunk(
                id=chunk['id'],
                content=chunk['content'],
                doc_title=chunk['doc_title'],
                category=chunk['category'],
                score=score
            ))
        
        return results
    
    def _calculate_score(self, query_freq: Counter, chunk: Dict) -> float:
        """Calculate relevance score between query and chunk"""
        chunk_freq = chunk['word_freq']
        
        # Calculate dot product
        score = 0
        for word, query_count in query_freq.items():
            chunk_count = chunk_freq.get(word, 0)
            if chunk_count > 0:
                # TF component
                tf = chunk_count / chunk['total_words'] if chunk['total_words'] > 0 else 0
                # Boost exact matches
                score += query_count * tf * 10
        
        # Boost for title matches
        title_words = set(chunk['doc_title'].lower().split())
        query_words = set(query_freq.keys())
        title_matches = len(title_words & query_words)
        score += title_matches * 5
        
        return score
    
    def format_context(
        self,
        chunks: List[SimpleRetrievedChunk],
        max_length: int = 2000
    ) -> str:
        """
        Format retrieved chunks into context for LLM
        
        Args:
            chunks: Retrieved chunks
            max_length: Maximum characters
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = []
        current_length = 0
        
        for chunk in chunks:
            formatted = f"""
[Sumber: {chunk.doc_title}]
{chunk.content}
---
"""
            
            if current_length + len(formatted) > max_length:
                break
            
            context_parts.append(formatted)
            current_length += len(formatted)
        
        return "\n".join(context_parts)
