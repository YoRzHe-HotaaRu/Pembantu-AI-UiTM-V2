"""
Text Chunker - Splits documents into manageable chunks for embedding
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    id: str
    content: str
    doc_id: str
    doc_title: str
    category: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]


class TextChunker:
    """Chunks documents into smaller pieces for embedding"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: List[str] = None
    ):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to use for splitting (in priority order)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ['\n## ', '\n### ', '\n\n', '\n', '. ', ' ']
    
    def chunk_document(self, document) -> List[TextChunk]:
        """
        Split a document into chunks
        
        Args:
            document: Document object from DocumentLoader
            
        Returns:
            List of TextChunk objects
        """
        text = document.content
        
        # If text is short enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [TextChunk(
                id=f"{document.id}#0",
                content=text,
                doc_id=document.id,
                doc_title=document.title,
                category=document.category,
                chunk_index=0,
                total_chunks=1,
                metadata=document.metadata
            )]
        
        # Split text into chunks
        chunks = self._split_text(text)
        
        # Create TextChunk objects
        text_chunks = []
        for i, chunk_text in enumerate(chunks):
            text_chunks.append(TextChunk(
                id=f"{document.id}#{i}",
                content=chunk_text,
                doc_id=document.id,
                doc_title=document.title,
                category=document.category,
                chunk_index=i,
                total_chunks=len(chunks),
                metadata=document.metadata
            ))
        
        return text_chunks
    
    def chunk_documents(self, documents: List) -> List[TextChunk]:
        """Chunk multiple documents"""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive character splitting
        """
        chunks = []
        
        # Start with the whole text
        remaining = text
        
        while remaining:
            if len(remaining) <= self.chunk_size:
                chunks.append(remaining.strip())
                break
            
            # Find the best split point
            split_point = self._find_split_point(remaining)
            
            # Extract chunk
            chunk = remaining[:split_point].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            remaining = remaining[split_point - self.chunk_overlap:]
        
        return chunks
    
    def _find_split_point(self, text: str) -> int:
        """
        Find the best point to split the text
        Prefers splitting at semantic boundaries (headers, paragraphs, sentences)
        """
        # If text is already small enough, return its length
        if len(text) <= self.chunk_size:
            return len(text)
        
        # Look for the best separator within chunk_size
        search_text = text[:self.chunk_size]
        
        for separator in self.separators:
            # Find the last occurrence of the separator
            pos = search_text.rfind(separator)
            if pos != -1:
                # Split after the separator
                return pos + len(separator)
        
        # If no good separator found, split at chunk_size
        return self.chunk_size
