"""
Image Handler - Manages image assets and their retrieval
Maps image references to file paths for display in responses
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ImageAsset:
    """Represents an image asset"""
    id: str
    filename: str
    path: str
    category: str
    description: str
    tags: List[str]
    url_path: str


class ImageHandler:
    """
    Manages image assets in the knowledge base
    Provides image retrieval based on text queries
    """
    
    def __init__(self, assets_path: str = "knowledge_base/assets"):
        self.assets_path = Path(assets_path)
        self.images: List[ImageAsset] = []
        self._image_index: Dict[str, List[str]] = {}  # tag -> image_ids
    
    def load_images(self):
        """Scan and index all images in the assets folder"""
        self.images = []
        self._image_index = {}
        
        if not self.assets_path.exists():
            print(f"Assets path not found: {self.assets_path}")
            return
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'}
        
        # Scan all subdirectories
        for category_dir in self.assets_path.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                
                # Look for metadata file
                metadata_file = category_dir / '_metadata.json'
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                # Scan images
                for img_path in category_dir.rglob('*'):
                    if img_path.is_file() and img_path.suffix.lower() in image_extensions:
                        self._add_image(img_path, category, metadata)
        
        print(f"Loaded {len(self.images)} images from assets")
    
    def _add_image(self, img_path: Path, category: str, metadata: Dict):
        """Add an image to the index"""
        img_id = f"{category}/{img_path.name}"
        
        # Get description from metadata if available
        img_metadata = metadata.get(img_path.name, {})
        description = img_metadata.get('description', img_path.stem.replace('_', ' '))
        tags = img_metadata.get('tags', [])
        
        # Create URL path (relative to static folder)
        url_path = f"/static/kb_assets/{category}/{img_path.name}"
        
        image = ImageAsset(
            id=img_id,
            filename=img_path.name,
            path=str(img_path),
            category=category,
            description=description,
            tags=tags,
            url_path=url_path
        )
        
        self.images.append(image)
        
        # Index by tags
        all_tags = tags + [category, img_path.stem.lower()]
        for tag in all_tags:
            if tag not in self._image_index:
                self._image_index[tag] = []
            self._image_index[tag].append(img_id)
    
    def search_images(self, query: str, limit: int = 5) -> List[ImageAsset]:
        """
        Search for images matching a query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching ImageAsset objects
        """
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Score each image
        image_scores = {}
        
        for img in self.images:
            score = 0
            
            # Check description
            desc_lower = img.description.lower()
            if query_lower in desc_lower:
                score += 3
            
            # Check tags
            for tag in img.tags:
                tag_lower = tag.lower()
                for word in query_words:
                    if word in tag_lower:
                        score += 2
            
            # Check category
            if any(word in img.category.lower() for word in query_words):
                score += 1
            
            # Check filename
            if any(word in img.filename.lower() for word in query_words):
                score += 1
            
            if score > 0:
                image_scores[img.id] = (img, score)
        
        # Sort by score and return top results
        sorted_results = sorted(
            image_scores.values(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
        
        return [img for img, score in sorted_results]
    
    def get_image_by_id(self, image_id: str) -> Optional[ImageAsset]:
        """Get a specific image by ID"""
        for img in self.images:
            if img.id == image_id:
                return img
        return None
    
    def get_images_by_category(self, category: str) -> List[ImageAsset]:
        """Get all images in a category"""
        return [img for img in self.images if img.category == category]
    
    def suggest_images_for_context(self, context: str) -> List[ImageAsset]:
        """
        Suggest relevant images based on context text
        
        Args:
            context: Text context from retrieved documents
            
        Returns:
            List of suggested ImageAsset objects
        """
        context_lower = context.lower()
        suggestions = []
        
        # Check each image for relevance
        for img in self.images:
            relevance = 0
            
            # Check if image tags appear in context
            for tag in img.tags:
                if tag.lower() in context_lower:
                    relevance += 1
            
            # Check if category appears
            if img.category.lower() in context_lower:
                relevance += 1
            
            if relevance > 0:
                suggestions.append((img, relevance))
        
        # Sort by relevance
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return [img for img, _ in suggestions[:3]]  # Top 3 suggestions
    
    def create_image_reference(self, image: ImageAsset) -> str:
        """Create a markdown reference for an image"""
        return f"![{image.description}]({image.url_path})"
    
    def get_all_categories(self) -> List[str]:
        """Get list of all image categories"""
        categories = set(img.category for img in self.images)
        return sorted(list(categories))
