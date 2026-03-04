# RAG System Usage Guide

This guide explains how to use and maintain the Retrieval-Augmented Generation (RAG) system in your UITM Chatbot.

## Overview

The RAG system allows the chatbot to pull information from an organized folder structure containing documents, images, and assets. This enables the AI to provide accurate, sourced answers based on your curated knowledge base.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**For Lightweight Mode (Default):**
- No additional dependencies required!
- Uses simple keyword matching

**For Advanced Mode (Optional):**
- `chromadb` - Vector database for similarity search
- `sentence-transformers` - For generating text embeddings
- `PyPDF2` - For parsing PDF files
- `numpy` - For numerical operations
- Requires ~8GB RAM and downloads ~80MB model

### 2. Choose Your Mode

The RAG system has **two modes**:

#### Lightweight Mode (Default) ✅ Recommended
- Uses keyword-based TF-IDF matching
- **Fast startup** - no model downloads
- **Low memory** - works on any system
- **Good accuracy** for most queries

#### Advanced Mode (Optional)
- Uses semantic embeddings + keyword matching
- **Better accuracy** for complex queries
- **Slower startup** - downloads embedding model
- **Higher memory** - needs 8GB+ RAM

**To enable Advanced Mode**, add to `.env`:
```env
RAG_USE_ADVANCED=true
```

### 3. Add Documents to Knowledge Base

Place your documents in the appropriate folders:

```
knowledge_base/
├── 01-academic/        # Academic programs, courses
├── 02-admissions/      # Entry requirements, applications
├── 03-campus/          # Campus facilities, locations
├── 04-administrative/  # Contacts, forms, policies
├── 05-policies/        # University rules and guidelines
└── assets/             # Images, diagrams, maps
```

### 4. Run the Application

```bash
python app.py
```

The RAG system will automatically:
- Load all documents from the knowledge base
- **Lightweight mode**: Build keyword index (fast)
- **Advanced mode**: Create embeddings (slower, memory intensive)
- Make them available for retrieval

**First run:**
- Lightweight: ~2-5 seconds
- Advanced: ~30-60 seconds (downloads model)

## Document Format Guidelines

### Markdown (.md) - Recommended

Best for text-heavy content with structure.

```markdown
# Document Title

## Section Heading

Your content here...

- Bullet points
- Are supported

| Tables | Work | Too |
|--------|------|-----|
| Yes    | They | Do  |
```

### JSON (.json)

Best for structured data like contacts, schedules, etc.

```json
{
  "title": "Document Title",
  "data": [
    {"name": "Item 1", "value": "..."},
    {"name": "Item 2", "value": "..."}
  ]
}
```

### Plain Text (.txt)

Simple text files work too.

```
Title

Your content here...
```

### PDF (.pdf)

PDF files are supported but require PyPDF2. Text extraction quality varies.

## How It Works

### At Startup

1. **Document Loading** - All supported files are read from knowledge base
2. **Text Chunking** - Large documents are split into smaller chunks (~500 chars)
3. **Embedding Generation** - Each chunk is converted to a vector
4. **Vector Storage** - Embeddings are stored in ChromaDB for fast retrieval

### At Query Time

1. User asks a question
2. Query is converted to embedding vector
3. Vector database searches for similar chunks
4. Top 5 most relevant chunks are retrieved
5. Retrieved context is injected into the system prompt
6. AI generates answer based on context

## API Endpoints

### Search Knowledge Base

```bash
GET /api/knowledge/search?q=<query>&top_k=5&category=<category>
```

Example:
```bash
curl "http://localhost:5000/api/knowledge/search?q=library+hours"
```

Response:
```json
{
  "query": "library hours",
  "results": [
    {
      "id": "03-campus/facilities_library.md#0",
      "content": "Waktu Operasi: Isnin - Khamis: 8:00 pagi...",
      "doc_title": "Kemudahan Perpustakaan UITM",
      "category": "03-campus",
      "relevance": 0.89
    }
  ],
  "total": 1
}
```

### Get Categories

```bash
GET /api/knowledge/categories
```

### Get Statistics

```bash
GET /api/knowledge/stats
```

Response:
```json
{
  "total_chunks": 45,
  "total_documents": 8,
  "categories": {
    "03-campus": 20,
    "04-administrative": 15,
    "02-admissions": 10
  }
}
```

### Reload Knowledge Base

```bash
POST /api/knowledge/reload
```

Use this after adding new documents.

### Search Images

```bash
GET /api/images/search?q=<query>&limit=5
```

## Configuration

Add these to your `.env` file:

```env
# Enable/disable RAG system
ENABLE_RAG=true

# Number of chunks to retrieve per query (default: 5)
RAG_TOP_K=5

# Advanced mode with embeddings (requires 8GB+ RAM)
# Set to 'true' ONLY if you have sufficient memory
RAG_USE_ADVANCED=false
```

### Mode Comparison

| Feature | Lightweight Mode | Advanced Mode |
|---------|------------------|---------------|
| **Algorithm** | Keyword TF-IDF | Semantic + Keyword |
| **Startup Time** | ~2 seconds | ~30-60 seconds |
| **Memory Usage** | ~50MB | ~2-4GB |
| **Dependencies** | None | 4 packages |
| **Query Speed** | Fast | Fast |
| **Accuracy** | Good | Better |
| **Natural Language** | Moderate | Excellent |

**Recommendation:** Start with Lightweight Mode. Only enable Advanced Mode if:
- You have 8GB+ RAM
- You need better natural language understanding
- Your queries are complex and semantic
- Startup time is not critical

## Folder Organization Best Practices

### 1. Use Descriptive Filenames

```
✅ good:
  entry_requirements_degree.md
  important_contacts.json
  facilities_library.md

❌ bad:
  doc1.md
  info.txt
  file.pdf
```

### 2. Organize by Topic

```
knowledge_base/
├── 03-campus/
│   ├── facilities_library.md      # Library info
│   ├── facilities_sports.md       # Sports facilities
│   ├── shah_alam_main.md          # Main campus info
│   └── transport.md               # Transportation
```

### 3. Keep Documents Focused

Each document should cover one topic thoroughly rather than many topics briefly.

### 4. Use Headers for Structure

```markdown
# Main Title

## Subsection 1
Content...

## Subsection 2
More content...
```

## Adding New Documents

1. **Create the file** in the appropriate folder
2. **Use proper formatting** (Markdown headers, etc.)
3. **Save the file**
4. **Reload the knowledge base** using the API or restart the app

Example workflow:

```bash
# 1. Create new document
cat > knowledge_base/03-campus/hostel_info.md << 'EOF'
# Maklumat Kolej Kediaman

## Kolej Tun Dr. Ismail (KK1)

Kolej kediaman utama untuk pelajar tahun pertama...

Waktu Operasi Pejabat:
- Isnin-Jumaat: 8:00 pagi - 5:00 petang
- Sabtu: 8:00 pagi - 1:00 tengah hari

## Peraturan
1. Pelajar mesti berdaftar
2. Patuh peraturan disiplin
EOF

# 2. Reload knowledge base
curl -X POST http://localhost:5000/api/knowledge/reload
```

## Troubleshooting

### Issue: System crashes during startup / Freezes at "Chunking documents"

**Cause:** The advanced embedding mode uses too much memory.

**Solution:** Use Lightweight Mode (default)

1. Make sure `.env` does NOT have `RAG_USE_ADVANCED=true`
2. Or explicitly set:
   ```env
   RAG_USE_ADVANCED=false
   ```
3. Restart the application

The lightweight mode uses simple keyword matching and works on any system.

### Issue: RAG not retrieving documents

**Check:**
1. Documents are in supported format (.md, .json, .txt, .pdf)
2. Documents are in the correct folder structure
3. ENABLE_RAG is set to true in .env
4. Check server logs for indexing errors

### Issue: First startup is slow (Advanced Mode)

**Solution:** This is normal for Advanced Mode. The system needs to:
1. Download the embedding model (~80MB)
2. Generate embeddings for all documents

**To speed up:** Switch to Lightweight Mode which is instant.

### Issue: Out of memory

**Solutions:**

**Option 1 - Use Lightweight Mode (Recommended):**
```env
RAG_USE_ADVANCED=false
```

**Option 2 - Reduce batch size (Advanced Mode only):**
Edit `rag/rag_manager.py`:
```python
batch_size = 16  # Reduce from 32
```

**Option 3 - Reduce chunk size:**
```python
chunk_size=300  # Reduce from 500
```

### Issue: Search results not relevant

**Solutions:**
1. **Lightweight Mode:** Use exact keywords that appear in documents
2. **Advanced Mode:** Better for natural language queries
3. Improve document titles and headings
4. Add more descriptive content
5. Check that documents are properly chunked

### Issue: "ImportError: No module named 'chromadb'" or similar

**Solution:** These are optional dependencies for Advanced Mode.

**Option 1:** Use Lightweight Mode (default) - no additional packages needed

**Option 2:** Install optional dependencies:
```bash
pip install chromadb sentence-transformers PyPDF2 numpy
```

## Performance Tips

1. **Keep documents under 5000 words** - Large documents are split anyway
2. **Use specific filenames** - Helps with debugging
3. **Regular maintenance** - Remove outdated documents
4. **Cache directory** - Don't delete `rag_cache/` folder

## Maintenance

### Regular Tasks

**Monthly:**
- Review and update outdated documents
- Check search logs for failed queries
- Add new documents based on common questions

**Quarterly:**
- Full reindex: `POST /api/knowledge/reload`
- Clean up unused documents
- Optimize document structure

### Backup

Important folders to backup:
- `knowledge_base/` - Your documents
- `rag_cache/` - Generated embeddings (optional, can be regenerated)

```bash
# Backup script
 tar -czf kb_backup_$(date +%Y%m%d).tar.gz knowledge_base/ rag_cache/
```

## Advanced Usage

### Custom Chunking

Edit `rag/chunker.py` to adjust chunk size:

```python
chunker = TextChunker(
    chunk_size=300,      # Smaller chunks
    chunk_overlap=50     # Less overlap
)
```

### Custom Retrieval

Edit `rag/retriever.py` to adjust search behavior:

```python
retriever = HybridRetriever(
    semantic_weight=0.8,    # Prefer semantic search
    keyword_weight=0.2,
    keyword_boost=0.3       # Boost exact matches
)
```

### Image Metadata

Add `_metadata.json` to image folders:

```json
{
  "campus_map.png": {
    "description": "Peta Kampus Shah Alam",
    "tags": ["peta", "kampus", "shah alam", "lokasi"]
  }
}
```

## Examples

### Example 1: Library Hours Query

```
User: "Berapa waktu perpustakaan buka?"

Retrieved Context:
[Sumber: Kemudahan Perpustakaan UITM]
Waktu Operasi Semester:
- Isnin - Khamis: 8:00 pagi - 10:00 malam
- Jumaat: 8:00 pagi - 12:00 tengah hari, 2:45 petang - 10:00 malam
---

AI Response:
Perpustakaan Tun Abdul Razak (PTAR) beroperasi pada waktu berikut:

Semasa Semester:
• Isnin - Khamis: 8:00 pagi - 10:00 malam
• Jumaat: 8:00 pagi - 12:00 tengah hari, 2:45 petang - 10:00 malam
• Sabtu: 9:00 pagi - 5:00 petang
• Ahad: 10:00 pagi - 5:00 petang
```

### Example 2: Contact Information

```
User: "Nombor telefon perpustakaan?"

Retrieved Context:
[Sumber: Hubungan Penting UITM]
Perpustakaan Tun Abdul Razak (PTAR)
Telefon: 03-5544 3740
Emel: ptar@uitm.edu.my
---

AI Response:
Anda boleh menghubungi Perpustakaan Tun Abdul Razak (PTAR) di:

📞 Telefon: 03-5544 3740
📧 Emel: ptar@uitm.edu.my
```

## Support

For issues or questions:
1. Check the server logs for errors
2. Verify document formatting
3. Test with the API endpoints directly
4. Review this guide for common issues

---

**Note:** The RAG system is designed to enhance the chatbot's knowledge without requiring code changes. Simply add documents to the knowledge base and reload!
