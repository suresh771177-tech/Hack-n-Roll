import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import pandas as pd
from datetime import datetime
import uuid

# Essential imports only
try:
    import fitz  # PyMuPDF - most reliable PDF reader
    PDF_AVAILABLE = True
except ImportError:
    print("Install PyMuPDF: pip install PyMuPDF")
    PDF_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
    import faiss
    EMBEDDING_AVAILABLE = True
except ImportError:
    print("Install: pip install sentence-transformers faiss-cpu")
    EMBEDDING_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Install: pip install google-generativeai")
    GEMINI_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
except ImportError:
    print("Install PaddleOCR: pip install paddleocr")
    OCR_AVAILABLE = False

# Hardcoded API key - Replace with your actual API key

class ChatHistory:
    """Enhanced chat history management with metadata"""
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.session_id = str(uuid.uuid4())[:8]
        self.chat_entries = []
        self.created_at = datetime.now()
    
    def add_entry(self, question: str, answer: str, sources: str, chunk_count: int, pdf_sources: List[str] = None):
        """Add a chat entry with full metadata"""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunk_count": chunk_count,
            "pdf_sources": pdf_sources or [],
            "collection_name": self.collection_name,
            "session_id": self.session_id
        }
        self.chat_entries.append(entry)
    
    def get_formatted_history(self, format_type: str = "txt") -> str:
        """Generate formatted chat history for download"""
        if format_type == "txt":
            return self._format_as_text()
        elif format_type == "markdown":
            return self._format_as_markdown()
        else:
            return self._format_as_text()
    
    def _format_as_text(self) -> str:
        """Format as clean text file"""
        header = f"""
STUDYAI CHAT HISTORY
====================================================
Collection: {self.collection_name}
Session ID: {self.session_id}
Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}
Total Questions: {len(self.chat_entries)}
====================================================

"""
        
        content = header
        for i, entry in enumerate(self.chat_entries, 1):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            content += f"""
QUESTION {i} - {timestamp}
{'-' * 50}
Q: {entry['question']}

ANSWER:
{entry['answer']}

SOURCES: {entry['sources']}
PDF SOURCES: {', '.join(entry['pdf_sources']) if entry['pdf_sources'] else 'N/A'}
CHUNKS USED: {entry['chunk_count']}

{'=' * 80}
"""
        
        return content
    
    def _format_as_markdown(self) -> str:
        """Format as markdown file"""
        header = f"""# StudyAI Chat History

**Collection:** {self.collection_name}  
**Session ID:** {self.session_id}  
**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}  
**Total Questions:** {len(self.chat_entries)}

---

"""
        
        content = header
        for i, entry in enumerate(self.chat_entries, 1):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            content += f"""
## Question {i} - {timestamp}

**Question:** {entry['question']}

**Answer:**
{entry['answer']}

**Metadata:**
- Sources: {entry['sources']}
- PDF Sources: {', '.join(entry['pdf_sources']) if entry['pdf_sources'] else 'N/A'}
- Chunks Used: {entry['chunk_count']}

---
"""
        
        return content
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "collection_name": self.collection_name,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "chat_entries": self.chat_entries
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Load from dictionary"""
        instance = cls(data["collection_name"])
        instance.session_id = data["session_id"]
        instance.created_at = datetime.fromisoformat(data["created_at"])
        instance.chat_entries = data["chat_entries"]
        return instance

class SimpleRAGPipeline:
    """Enhanced RAG pipeline with IBM Granite encoder and cross-encoder reranking"""
    
    def __init__(self, name: str, api_key: str = None):
        self.name = name
        self.api_key = api_key or GEMINI_API_KEY
        self.chunks = []
        self.chunk_metadata = []  # Store metadata for each chunk
        self.embeddings = None
        self.index = None
        self.is_ready = False
        self.processing_stats = {}
        self.chat_history = ChatHistory(name)
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize embedding, cross-encoder, OCR and generation models"""
        try:
            if EMBEDDING_AVAILABLE:
                # Use IBM Granite embedding model
                try:
                    self.embedding_model = SentenceTransformer('ibm/granite-embedding-english-r2')
                    logging.info("IBM Granite embedding model loaded")
                except Exception as e:
                    logging.warning(f"Failed to load IBM Granite model: {e}, falling back to MiniLM")
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logging.info("Fallback embedding model loaded")
                
                # Initialize cross-encoder for reranking
                try:
                    self.cross_encoder = CrossEncoder('BAAI/bge-reranker-base')
                    logging.info("Cross-encoder reranker loaded")
                except Exception as e:
                    logging.warning(f"Failed to load cross-encoder: {e}")
                    self.cross_encoder = None
            
            if OCR_AVAILABLE:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
                logging.info("OCR model loaded")
            
            if GEMINI_AVAILABLE and self.api_key:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logging.info("Gemini model initialized")
            
        except Exception as e:
            logging.error(f"Model initialization error: {e}")
    
    def extract_text_with_ocr(self, pdf_path: str, filename: str) -> List[Dict]:
        """
        Run OCR on each page of the PDF with metadata
        Returns list of dictionaries with text and metadata
        """
        if not OCR_AVAILABLE:
            return [{"text": "OCR not available - install paddleocr", "source_file": filename, "page": 1, "extraction_method": "error"}]
        
        try:
            doc = fitz.open(pdf_path)
            sections = []
            
            for page_num in range(len(doc)):
                try:
                    # Render page to image with high DPI for better OCR
                    pix = doc[page_num].get_pixmap(dpi=300)
                    img_path = f"temp_page_{page_num}_{os.getpid()}.png"
                    pix.save(img_path)
                    
                    # Run OCR
                    result = self.ocr.ocr(img_path, cls=True)
                    
                    page_lines = []
                    if result and result[0]:
                        for line in result[0]:
                            if line[1]:
                                text = line[1][0].strip()
                                confidence = line[1][1]
                                # Only include text with reasonable confidence
                                if confidence > 0.5 and len(text) > 2:
                                    page_lines.append(text)
                    
                    if page_lines:
                        page_text = "\n".join(page_lines)
                        sections.append({
                            "text": page_text,
                            "source_file": filename,
                            "page": page_num + 1,
                            "extraction_method": "ocr",
                            "processed_at": datetime.now().isoformat()
                        })
                    
                    # Clean up temp image
                    if os.path.exists(img_path):
                        os.unlink(img_path)
                        
                except Exception as e:
                    logging.warning(f"OCR failed for page {page_num}: {e}")
                    continue
            
            doc.close()
            return sections
            
        except Exception as e:
            logging.error(f"OCR extraction failed: {e}")
            return []
    
    def detect_images_in_pdf(self, pdf_path: str) -> Dict:
        """
        Detect if PDF pages contain images that might have text.
        Returns analysis of image content per page.
        """
        image_analysis = {"pages_with_images": [], "total_images": 0, "image_rich_pages": []}
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get list of images on the page
                image_list = page.get_images()
                
                if image_list:
                    image_analysis["pages_with_images"].append(page_num + 1)
                    image_analysis["total_images"] += len(image_list)
                    
                    # Check if page is image-rich (more than 2 images or large images)
                    large_images = 0
                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image details
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)
                            
                            # Consider image large if width or height > 200 pixels
                            if pix.width > 200 or pix.height > 200:
                                large_images += 1
                            
                            pix = None  # Clean up
                        except:
                            continue
                    
                    # Mark as image-rich if more than 2 images or any large images
                    if len(image_list) > 2 or large_images > 0:
                        image_analysis["image_rich_pages"].append(page_num + 1)
            
            doc.close()
            
        except Exception as e:
            logging.warning(f"Image detection failed: {e}")
        
        return image_analysis
    
    def should_use_hybrid_extraction(self, pdf_path: str, text_content: List[Dict]) -> bool:
        """
        Determine if hybrid extraction (text + OCR) should be used.
        Returns True if the PDF has mixed content that needs OCR supplementation.
        """
        try:
            # Analyze image content
            image_analysis = self.detect_images_in_pdf(pdf_path)
            
            # Check text extraction quality
            total_text_length = sum(len(content["text"]) for content in text_content)
            avg_text_per_page = total_text_length / max(len(text_content), 1)
            
            # Decision criteria for hybrid extraction:
            # 1. PDF has images on multiple pages
            # 2. Some pages have very little text but images present
            # 3. Image-rich pages detected
            
            has_multiple_image_pages = len(image_analysis["pages_with_images"]) > 1
            has_image_rich_pages = len(image_analysis["image_rich_pages"]) > 0
            has_sparse_text = avg_text_per_page < 200  # Less than 200 chars per page on average
            
            should_hybrid = (has_multiple_image_pages and has_sparse_text) or has_image_rich_pages
            
            if should_hybrid:
                logging.info(f"Hybrid extraction recommended - Images on {len(image_analysis['pages_with_images'])} pages, "
                           f"avg text per page: {avg_text_per_page:.0f} chars")
            
            return should_hybrid
            
        except Exception as e:
            logging.warning(f"Hybrid extraction analysis failed: {e}")
            return False
    
    def extract_hybrid_content(self, pdf_path: str, filename: str) -> List[Dict]:
        """
        Extract content using hybrid approach with metadata
        """
        all_content = []
        
        try:
            doc = fitz.open(pdf_path)
            image_analysis = self.detect_images_in_pdf(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_number = page_num + 1
                
                # Always try text extraction first
                text_content = page.get_text().strip()
                
                # Check if this page needs OCR supplementation
                needs_ocr = (
                    page_number in image_analysis["image_rich_pages"] or
                    (page_number in image_analysis["pages_with_images"] and len(text_content) < 100)
                )
                
                if needs_ocr and OCR_AVAILABLE:
                    logging.info(f"Using hybrid extraction for page {page_number}")
                    
                    # Render page to image for OCR
                    pix = page.get_pixmap(dpi=300)
                    img_path = f"temp_hybrid_page_{page_number}_{os.getpid()}.png"
                    pix.save(img_path)
                    
                    try:
                        # Run OCR on the page
                        result = self.ocr.ocr(img_path, cls=True)
                        
                        ocr_lines = []
                        if result and result[0]:
                            for line in result[0]:
                                if line[1]:
                                    text = line[1][0].strip()
                                    confidence = line[1][1]
                                    if confidence > 0.6 and len(text) > 2:
                                        ocr_lines.append(text)
                        
                        # Combine text extraction with OCR
                        combined_content = []
                        
                        if text_content:
                            combined_content.append("[Text Extracted Content]")
                            combined_content.append(text_content)
                        
                        if ocr_lines:
                            combined_content.append("[OCR Extracted Content]")
                            combined_content.extend(ocr_lines)
                        
                        if combined_content:
                            page_content = "\n".join(combined_content)
                            all_content.append({
                                "text": page_content,
                                "source_file": filename,
                                "page": page_number,
                                "extraction_method": "hybrid",
                                "processed_at": datetime.now().isoformat()
                            })
                        
                        # Clean up temp image
                        if os.path.exists(img_path):
                            os.unlink(img_path)
                            
                    except Exception as e:
                        logging.warning(f"OCR failed for page {page_number}: {e}")
                        if text_content:
                            all_content.append({
                                "text": text_content,
                                "source_file": filename,
                                "page": page_number,
                                "extraction_method": "text",
                                "processed_at": datetime.now().isoformat()
                            })
                
                else:
                    # Use only text extraction
                    if text_content:
                        all_content.append({
                            "text": text_content,
                            "source_file": filename,
                            "page": page_number,
                            "extraction_method": "text",
                            "processed_at": datetime.now().isoformat()
                        })
            
            doc.close()
            return all_content
            
        except Exception as e:
            logging.error(f"Hybrid extraction failed: {e}")
            return []

    def extract_text_from_pdfs(self, pdf_files: List) -> List[Dict]:
        """Enhanced PDF text extraction with metadata tracking"""
        all_content = []
        extraction_stats = {"fitz_success": 0, "ocr_fallback": 0, "hybrid_extraction": 0, "failed": 0}
        
        if not PDF_AVAILABLE:
            return [{"text": f"Mock text from {pdf.name}", "source_file": pdf.name, "page": 1, "extraction_method": "mock"} for pdf in pdf_files]
        
        for pdf_file in pdf_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_path = tmp_file.name
                
                filename = pdf_file.name
                
                # First, try standard text extraction
                doc = fitz.open(tmp_path)
                text_content = []
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text().strip()
                    if text:
                        text_content.append({
                            "text": text,
                            "source_file": filename,
                            "page": page_num + 1,
                            "extraction_method": "text",
                            "processed_at": datetime.now().isoformat()
                        })
                
                doc.close()
                
                # Analyze if we need hybrid extraction
                if text_content and self.should_use_hybrid_extraction(tmp_path, text_content):
                    # Use hybrid extraction for mixed content
                    logging.info(f"Using hybrid extraction for {filename}")
                    hybrid_content = self.extract_hybrid_content(tmp_path, filename)
                    
                    if hybrid_content:
                        all_content.extend(hybrid_content)
                        extraction_stats["hybrid_extraction"] += 1
                        logging.info(f"Successfully extracted hybrid content from {filename}")
                    else:
                        # Fallback to simple text extraction
                        all_content.extend(text_content)
                        extraction_stats["fitz_success"] += 1
                
                elif text_content and sum(len(content["text"]) for content in text_content) > 100:
                    # Standard text extraction was sufficient
                    all_content.extend(text_content)
                    extraction_stats["fitz_success"] += 1
                    logging.info(f"Successfully extracted text from {filename} using standard extraction")
                
                else:
                    # Pure OCR fallback for image-only PDFs
                    logging.info(f"Standard extraction failed for {filename}, trying full OCR...")
                    ocr_content = self.extract_text_with_ocr(tmp_path, filename)
                    if ocr_content and ocr_content[0].get("text") != "OCR not available - install paddleocr":
                        all_content.extend(ocr_content)
                        extraction_stats["ocr_fallback"] += 1
                        logging.info(f"Successfully extracted text from {filename} using OCR")
                    else:
                        all_content.append({
                            "text": f"Failed to extract text from {filename}",
                            "source_file": filename,
                            "page": 1,
                            "extraction_method": "failed",
                            "processed_at": datetime.now().isoformat()
                        })
                        extraction_stats["failed"] += 1
                
                # Clean up
                os.unlink(tmp_path)
                
            except Exception as e:
                all_content.append({
                    "text": f"Error processing {pdf_file.name}: {str(e)}",
                    "source_file": pdf_file.name,
                    "page": 1,
                    "extraction_method": "error",
                    "processed_at": datetime.now().isoformat()
                })
                extraction_stats["failed"] += 1
                logging.error(f"Error processing {pdf_file.name}: {e}")
        
        self.processing_stats = extraction_stats
        return all_content
    
    def create_chunks(self, content_list: List[Dict], chunk_size: int = 800) -> Tuple[List[str], List[Dict]]:
        """Enhanced text chunking with metadata preservation"""
        chunks = []
        chunk_metadata = []
        
        for content in content_list:
            text = content["text"]
            metadata = {
                "source_file": content["source_file"],
                "page": content["page"],
                "extraction_method": content["extraction_method"],
                "processed_at": content["processed_at"]
            }
            
            # Clean and preprocess text
            text = text.replace('\n\n', ' [PARA] ').replace('\t', ' ')
            
            # Split by sentences (improved)
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + " "
                else:
                    # Save current chunk if it's substantial
                    if len(current_chunk.strip()) > 100:
                        chunks.append(current_chunk.strip())
                        chunk_metadata.append(metadata.copy())
                    current_chunk = sentence + " "
            
            # Add final chunk
            if len(current_chunk.strip()) > 100:
                chunks.append(current_chunk.strip())
                chunk_metadata.append(metadata.copy())
        
        return chunks, chunk_metadata
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings using IBM Granite model with error handling"""
        if not EMBEDDING_AVAILABLE or not hasattr(self, 'embedding_model'):
            # Return mock embeddings
            return np.random.rand(len(chunks), 768)  # Granite model has 768 dimensions
        
        try:
            embeddings = self.embedding_model.encode(
                chunks, 
                show_progress_bar=True,
                batch_size=16,  # Smaller batch size for larger model
                normalize_embeddings=True
            )
            logging.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logging.error(f"Embedding error: {e}")
            return np.random.rand(len(chunks), 768)
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index with error handling"""
        try:
            if EMBEDDING_AVAILABLE and embeddings is not None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(embeddings.astype('float32'))
                logging.info(f"Built FAISS index with {len(embeddings)} vectors")
            else:
                self.index = None
        except Exception as e:
            logging.error(f"Index building error: {e}")
            self.index = None
    
    def process_pdfs(self, pdf_files: List) -> Dict:
        """Enhanced main processing pipeline"""
        try:
            # Step 1: Extract text with metadata
            content_list = self.extract_text_from_pdfs(pdf_files)
            if not content_list:
                return {"success": False, "error": "No text extracted from PDFs"}
            
            # Step 2: Create chunks with metadata
            self.chunks, self.chunk_metadata = self.create_chunks(content_list)
            if not self.chunks:
                return {"success": False, "error": "No chunks created"}
            
            # Step 3: Create embeddings
            self.embeddings = self.create_embeddings(self.chunks)
            
            # Step 4: Build index
            self.build_index(self.embeddings)
            
            self.is_ready = True
            
            return {
                "success": True,
                "total_files": len(pdf_files),
                "total_chunks": len(self.chunks),
                "processing_stats": self.processing_stats,
                "message": "Processing completed successfully!"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Processing failed: {str(e)}"}
    
    def search_similar(self, question: str, top_k: int = 5) -> List[Dict]:
        """Enhanced similarity search with cross-encoder reranking"""
        if not self.is_ready or not self.chunks:
            return []
        
        try:
            if self.index is not None and hasattr(self, 'embedding_model'):
                # Initial retrieval with semantic search
                query_embedding = self.embedding_model.encode([question], normalize_embeddings=True)
                distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k * 2, len(self.chunks)))
                
                # Prepare candidates for reranking
                candidates = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks):
                        candidates.append({
                            'text': self.chunks[idx],
                            'distance': distances[0][i],
                            'index': idx,
                            'metadata': self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                        })
                
                # Rerank using cross-encoder if available
                if self.cross_encoder and len(candidates) > 1:
                    try:
                        pairs = [[question, candidate['text']] for candidate in candidates]
                        scores = self.cross_encoder.predict(pairs)
                        
                        # Combine with original scores
                        for i, candidate in enumerate(candidates):
                            candidate['rerank_score'] = scores[i]
                            candidate['combined_score'] = 0.3 * (1 - candidate['distance']) + 0.7 * scores[i]
                        
                        # Sort by combined score
                        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
                        logging.info("Applied cross-encoder reranking")
                    except Exception as e:
                        logging.warning(f"Reranking failed: {e}")
                
                return candidates[:top_k]
            else:
                # Mock search - return first few chunks with metadata
                mock_results = []
                for i in range(min(top_k, len(self.chunks))):
                    mock_results.append({
                        'text': self.chunks[i],
                        'distance': 0.5,
                        'index': i,
                        'metadata': self.chunk_metadata[i] if i < len(self.chunk_metadata) else {}
                    })
                return mock_results
                
        except Exception as e:
            logging.error(f"Search error: {e}")
            mock_results = []
            for i in range(min(top_k, len(self.chunks))):
                mock_results.append({
                    'text': self.chunks[i],
                    'distance': 0.5,
                    'index': i,
                    'metadata': self.chunk_metadata[i] if i < len(self.chunk_metadata) else {}
                })
            return mock_results
    
    def generate_answer(self, question: str, context_results: List[Dict]) -> str:
        """Enhanced answer generation using Gemini with metadata awareness"""
        try:
            if not hasattr(self, 'gemini_model') or not self.gemini_model:
                # Mock response with metadata
                sources_info = []
                for result in context_results[:2]:
                    metadata = result.get('metadata', {})
                    source = f"{metadata.get('source_file', 'Unknown')} (Page {metadata.get('page', 'N/A')})"
                    sources_info.append(source)
                
                return f"Based on your document collection '{self.name}' from sources: {', '.join(sources_info)}, here's what I found regarding '{question}': {' '.join([r['text'] for r in context_results[:2]])[:400]}... This is a comprehensive answer based on the relevant sections of your uploaded documents."
            
            # Create context with source attribution
            context_parts = []
            source_files = set()
            
            for i, result in enumerate(context_results[:3]):
                metadata = result.get('metadata', {})
                source_file = metadata.get('source_file', 'Unknown')
                page = metadata.get('page', 'N/A')
                
                source_files.add(source_file)
                context_parts.append(f"[Source: {source_file}, Page {page}]\n{result['text']}")
            
            context = "\n\n".join(context_parts)
            
            # Enhanced prompt with source awareness
            prompt = f"""You are an AI assistant analyzing documents. Based on the following context from the uploaded documents, provide a comprehensive and accurate answer to the user's question.

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Provide a detailed, well-structured answer based solely on the context above
- Reference specific sources when making claims (mention document names and pages when relevant)
- If the context doesn't contain enough information, acknowledge this limitation
- Use specific details and examples from the context when possible
- Format your response clearly with proper structure
- Be precise and avoid speculation

ANSWER:"""

            # Generate response
            response = self.gemini_model.generate_content(prompt)
            
            if response.text:
                return response.text
            else:
                return "I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return f"I encountered an error while generating the response. Please try again with a different question."
    
    def query(self, question: str) -> Dict:
        """Enhanced main query method with chat history tracking"""
        if not self.is_ready:
            return {
                "success": False,
                "answer": "Knowledge base not ready. Please process documents first."
            }
        
        try:
            # Search for relevant chunks with metadata
            relevant_results = self.search_similar(question, top_k=5)
            
            if not relevant_results:
                return {
                    "success": False,
                    "answer": "No relevant information found in the documents."
                }
            
            # Generate answer
            answer = self.generate_answer(question, relevant_results)
            
            # Extract source information
            source_files = set()
            for result in relevant_results:
                metadata = result.get('metadata', {})
                if metadata.get('source_file'):
                    source_files.add(metadata['source_file'])
            
            sources_text = f"Based on {len(relevant_results)} relevant sections from: {', '.join(source_files)}"
            
            # Add to chat history
            self.chat_history.add_entry(
                question=question,
                answer=answer,
                sources=sources_text,
                chunk_count=len(relevant_results),
                pdf_sources=list(source_files)
            )
            
            return {
                "success": True,
                "answer": answer,
                "sources": sources_text,
                "chunk_count": len(relevant_results),
                "pdf_sources": list(source_files)
            }
            
        except Exception as e:
            logging.error(f"Query error: {e}")
            return {
                "success": False,
                "answer": f"Error processing question: {str(e)}"
            }
    
    def get_chat_history_download(self, format_type: str = "txt") -> str:
        """Get formatted chat history for download"""
        return self.chat_history.get_formatted_history(format_type)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving"""
        return {
            "name": self.name,
            "chunks": self.chunks,
            "chunk_metadata": self.chunk_metadata,
            "is_ready": self.is_ready,
            "processing_stats": self.processing_stats,
            "chat_history": self.chat_history.to_dict(),
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict, api_key: str = None):
        """Load from dictionary"""
        instance = cls(data["name"], api_key)
        instance.chunks = data["chunks"]
        instance.chunk_metadata = data.get("chunk_metadata", [])
        instance.is_ready = data["is_ready"]
        instance.processing_stats = data.get("processing_stats", {})
        
        # Load chat history if available
        if "chat_history" in data:
            instance.chat_history = ChatHistory.from_dict(data["chat_history"])
        
        if data["embeddings"]:
            instance.embeddings = np.array(data["embeddings"])
            instance.build_index(instance.embeddings)
        
        return instance

# Enhanced database manager
class SimpleDatabaseManager:
    def __init__(self, db_dir: str = "simple_rag_dbs"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)
        
        # Create chat history directory
        self.chat_history_dir = self.db_dir / "chat_histories"
        self.chat_history_dir.mkdir(exist_ok=True)
    
    def save_database(self, rag_pipeline: SimpleRAGPipeline) -> bool:
        """Save RAG pipeline with enhanced metadata"""
        try:
            db_path = self.db_dir / f"{rag_pipeline.name}.json"
            
            # Add metadata
            data = rag_pipeline.to_dict()
            data.update({
                "created_at": str(pd.Timestamp.now()),
                "db_id": str(hash(rag_pipeline.name))[-8:],
                "version": "3.0"  # Updated version
            })
            
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Database saved: {rag_pipeline.name}")
            return True
            
        except Exception as e:
            logging.error(f"Save error: {e}")
            return False
    
    def load_database(self, db_name: str, api_key: str = None) -> Optional[SimpleRAGPipeline]:
        """Load RAG pipeline"""
        try:
            db_path = self.db_dir / f"{db_name}.json"
            
            if not db_path.exists():
                return None
            
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return SimpleRAGPipeline.from_dict(data, api_key)
            
        except Exception as e:
            logging.error(f"Load error: {e}")
            return None
    
    def save_chat_history(self, collection_name: str, chat_history: ChatHistory, format_type: str = "txt") -> str:
        """Save chat history as downloadable file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{collection_name}_{chat_history.session_id}_{timestamp}.{format_type}"
            filepath = self.chat_history_dir / filename
            
            content = chat_history.get_formatted_history(format_type)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return str(filepath)
            
        except Exception as e:
            logging.error(f"Chat history save error: {e}")
            return None
    
    def get_databases(self) -> List[Dict]:
        """Get list of available databases with enhanced metadata"""
        databases = []
        
        for db_file in self.db_dir.glob("*.json"):
            try:
                with open(db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Count PDF sources
                chunk_metadata = data.get("chunk_metadata", [])
                pdf_sources = set()
                for metadata in chunk_metadata:
                    if metadata.get("source_file"):
                        pdf_sources.add(metadata["source_file"])
                
                # Get chat history stats
                chat_data = data.get("chat_history", {})
                chat_entries = chat_data.get("chat_entries", [])
                
                databases.append({
                    "name": data["name"],
                    "path": str(db_file),
                    "created_at": data.get("created_at", "Unknown"),
                    "chunks": len(data.get("chunks", [])),
                    "pdf_sources": list(pdf_sources),
                    "pdf_count": len(pdf_sources),
                    "chat_sessions": 1 if chat_entries else 0,
                    "total_questions": len(chat_entries),
                    "db_id": data.get("db_id", str(hash(data["name"]))[-8:]),
                    "version": data.get("version", "1.0"),
                    "processing_stats": data.get("processing_stats", {}),
                    "file_size": db_file.stat().st_size
                })
                
            except Exception as e:
                logging.error(f"Error reading {db_file}: {e}")
                continue
        
        return sorted(databases, key=lambda x: x["created_at"], reverse=True)
    
    def delete_database(self, db_name: str) -> bool:
        """Delete database and associated chat histories"""
        try:
            db_path = self.db_dir / f"{db_name}.json"
            if db_path.exists():
                db_path.unlink()
                
                # Clean up associated chat history files
                for chat_file in self.chat_history_dir.glob(f"{db_name}_*.txt"):
                    chat_file.unlink()
                for chat_file in self.chat_history_dir.glob(f"{db_name}_*.md"):
                    chat_file.unlink()
                
                logging.info(f"Database deleted: {db_name}")
                return True
            return False
            
        except Exception as e:
            logging.error(f"Delete error: {e}")
            return False

# Enhanced installation checker
def check_dependencies():
    """Check and guide installation"""
    missing = []
    
    if not PDF_AVAILABLE:
        missing.append("PyMuPDF")
    if not EMBEDDING_AVAILABLE:
        missing.append("sentence-transformers faiss-cpu")
    if not GEMINI_AVAILABLE:
        missing.append("google-generativeai")
    if not OCR_AVAILABLE:
        missing.append("paddleocr")
    
    return missing

def install_guide():
    """Print installation guide"""
    missing = check_dependencies()
    
    if missing:
        print("\n🚨 Missing Dependencies:")
        print("Run these commands to install:")
        print("pip install PyMuPDF sentence-transformers faiss-cpu google-generativeai paddleocr streamlit")
        print("\nFor GPU support (optional):")
        print("pip install faiss-gpu torch")
        print("\nFor enhanced models:")
        print("# IBM Granite embedding model will be downloaded automatically")
        print("# Cross-encoder reranker will be downloaded automatically")
        return False
    
    print("✅ All dependencies available!")
    return True