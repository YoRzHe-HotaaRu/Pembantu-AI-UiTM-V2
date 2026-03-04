"""
Document Loader - Loads and parses documents from knowledge base folders
Supports: Markdown, JSON, TXT, PDF
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a loaded document with metadata"""
    id: str
    content: str
    title: str
    source: str
    category: str
    file_type: str
    metadata: Dict[str, Any]


class DocumentLoader:
    """Loads documents from organized folder structure"""
    
    SUPPORTED_EXTENSIONS = {'.md', '.json', '.txt', '.pdf'}
    
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.documents: List[Document] = []
        
    def load_all(self) -> List[Document]:
        """Load all documents from knowledge base folders"""
        self.documents = []
        
        if not self.knowledge_base_path.exists():
            print(f"Knowledge base path not found: {self.knowledge_base_path}")
            return self.documents
        
        # Scan all subdirectories
        for category_dir in sorted(self.knowledge_base_path.iterdir()):
            if category_dir.is_dir() and not category_dir.name.startswith('_'):
                category = category_dir.name
                self._load_category(category_dir, category)
        
        print(f"Loaded {len(self.documents)} documents from knowledge base")
        return self.documents
    
    def _load_category(self, category_path: Path, category: str):
        """Load all documents from a category folder"""
        for file_path in category_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self._parse_file(file_path, category)
                    if doc:
                        self.documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    def _parse_file(self, file_path: Path, category: str) -> Optional[Document]:
        """Parse a single file based on its type"""
        file_type = file_path.suffix.lower()
        doc_id = f"{category}/{file_path.relative_to(self.knowledge_base_path / category)}"
        
        if file_type == '.md':
            return self._parse_markdown(file_path, doc_id, category)
        elif file_type == '.json':
            return self._parse_json(file_path, doc_id, category)
        elif file_type == '.txt':
            return self._parse_text(file_path, doc_id, category)
        elif file_type == '.pdf':
            return self._parse_pdf(file_path, doc_id, category)
        
        return None
    
    def _parse_markdown(self, file_path: Path, doc_id: str, category: str) -> Document:
        """Parse Markdown file"""
        content = file_path.read_text(encoding='utf-8')
        
        # Extract title from first H1 or filename
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem.replace('_', ' ').title()
        
        # Clean content (remove HTML comments, etc.)
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        return Document(
            id=doc_id,
            content=content,
            title=title,
            source=str(file_path),
            category=category,
            file_type='markdown',
            metadata={
                'word_count': len(content.split()),
                'line_count': len(content.split('\n'))
            }
        )
    
    def _parse_json(self, file_path: Path, doc_id: str, category: str) -> Document:
        """Parse JSON file - expects specific structure or converts to text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # If JSON has 'content' or 'data' field, use that
        if isinstance(data, dict):
            title = data.get('title', file_path.stem.replace('_', ' ').title())
            
            # Try to extract meaningful content
            if 'content' in data:
                content = data['content']
            elif 'data' in data:
                content = json.dumps(data['data'], indent=2, ensure_ascii=False)
            else:
                # Convert entire JSON to readable text
                content = self._json_to_text(data)
        else:
            title = file_path.stem.replace('_', ' ').title()
            content = json.dumps(data, indent=2, ensure_ascii=False)
        
        return Document(
            id=doc_id,
            content=content,
            title=title,
            source=str(file_path),
            category=category,
            file_type='json',
            metadata={'original_data': data}
        )
    
    def _json_to_text(self, data: Any, indent: int = 0) -> str:
        """Convert JSON structure to readable text"""
        lines = []
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(self._json_to_text(item, indent))
                else:
                    lines.append(f"{prefix}- {item}")
        
        return '\n'.join(lines)
    
    def _parse_text(self, file_path: Path, doc_id: str, category: str) -> Document:
        """Parse plain text file"""
        content = file_path.read_text(encoding='utf-8')
        
        return Document(
            id=doc_id,
            content=content,
            title=file_path.stem.replace('_', ' ').title(),
            source=str(file_path),
            category=category,
            file_type='text',
            metadata={'word_count': len(content.split())}
        )
    
    def _parse_pdf(self, file_path: Path, doc_id: str, category: str) -> Optional[Document]:
        """Parse PDF file - requires PyPDF2"""
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            return Document(
                id=doc_id,
                content=text.strip(),
                title=file_path.stem.replace('_', ' ').title(),
                source=str(file_path),
                category=category,
                file_type='pdf',
                metadata={
                    'page_count': len(pdf_reader.pages),
                    'word_count': len(text.split())
                }
            )
        except ImportError:
            print(f"PyPDF2 not installed, skipping PDF: {file_path}")
            return None
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return None
    
    def get_documents_by_category(self, category: str) -> List[Document]:
        """Get all documents in a specific category"""
        return [doc for doc in self.documents if doc.category == category]
    
    def search_by_keyword(self, keyword: str) -> List[Document]:
        """Simple keyword search across all documents"""
        keyword_lower = keyword.lower()
        results = []
        
        for doc in self.documents:
            if (keyword_lower in doc.content.lower() or 
                keyword_lower in doc.title.lower()):
                results.append(doc)
        
        return results
