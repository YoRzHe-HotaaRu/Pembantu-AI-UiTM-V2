"""
Embedding Engine - Creates vector representations of text using lightweight models
Uses sentence-transformers for local embedding generation
"""

import numpy as np
from typing import List, Optional
import hashlib
import json
from pathlib import Path


class EmbeddingEngine:
    """Generates embeddings for text chunks"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str = "rag_cache",
        use_cache: bool = True
    ):
        """
        Initialize embedding engine
        
        Args:
            model_name: Name of the sentence-transformers model
            cache_dir: Directory to cache embeddings
            use_cache: Whether to use caching
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.model = None
        self._embedding_cache = {}
        
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Lazy load the embedding model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                print("Embedding model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        self._load_model()
        
        # Check cache first
        if self.use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Cache result
        if self.use_cache:
            self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        self._load_model()
        
        # Check cache for each text
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)
        
        if self.use_cache:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._embedding_cache:
                    results[i] = self._embedding_cache[cache_key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                show_progress_bar=len(uncached_texts) > 10
            )
            
            # Store in results and cache
            for idx, text, embedding in zip(uncached_indices, uncached_texts, embeddings):
                results[idx] = embedding
                if self.use_cache:
                    cache_key = self._get_cache_key(text)
                    self._embedding_cache[cache_key] = embedding
        
        return np.array(results)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        self._load_model()
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def save_cache(self):
        """Save embedding cache to disk"""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / "embeddings_cache.json"
        # Convert numpy arrays to lists for JSON serialization
        serializable_cache = {
            k: v.tolist() for k, v in self._embedding_cache.items()
        }
        
        with open(cache_file, 'w') as f:
            json.dump(serializable_cache, f)
    
    def load_cache(self):
        """Load embedding cache from disk"""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / "embeddings_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                serializable_cache = json.load(f)
            
            # Convert lists back to numpy arrays
            self._embedding_cache = {
                k: np.array(v) for k, v in serializable_cache.items()
            }
