"""
OCR module for extracting text from scanned PDFs and image-based documents.
Uses Tesseract OCR with PIL for image preprocessing.
"""

import os
import tempfile
from typing import List, Optional
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

from langchain_core.documents import Document


class OCRProcessor:
    """Handles OCR processing for scanned documents and images."""
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR processor.
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
        """
        if not HAS_OCR:
            raise ImportError(
                "OCR dependencies not installed. "
                "Install with: pip install pytesseract pillow pymupdf"
            )
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Test if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                f"Tesseract not found. Please install Tesseract OCR. Error: {e}"
            )
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize if image is too small (OCR works better on larger images)
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = max(1000 / width, 1000 / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def extract_text_from_image(self, image: Image.Image, lang: str = 'eng') -> str:
        """
        Extract text from a PIL Image using OCR.
        
        Args:
            image: PIL Image object
            lang: Language for OCR (default: 'eng')
            
        Returns:
            Extracted text
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Configure OCR
            custom_config = r'--oem 3 --psm 6'
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image,
                lang=lang,
                config=custom_config
            )
            
            return text.strip()
            
        except Exception as e:
            print(f"OCR error: {str(e)}")
            return ""
    
    def is_pdf_scanned(self, pdf_path: Path, sample_pages: int = 3) -> bool:
        """
        Check if a PDF contains scanned images by examining text content.
        
        Args:
            pdf_path: Path to PDF file
            sample_pages: Number of pages to sample for detection
            
        Returns:
            True if PDF appears to be scanned
        """
        try:
            doc = fitz.open(str(pdf_path))
            
            total_pages = len(doc)
            pages_to_check = min(sample_pages, total_pages)
            
            text_length = 0
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text()
                text_length += len(text.strip())
            
            doc.close()
            
            # If very little text found, likely scanned
            avg_text_per_page = text_length / pages_to_check
            return avg_text_per_page < 100  # Threshold for "scanned"
            
        except Exception as e:
            print(f"Error checking PDF: {str(e)}")
            return False
    
    def extract_text_from_pdf_ocr(self, pdf_path: Path, lang: str = 'eng') -> List[Document]:
        """
        Extract text from scanned PDF using OCR.
        
        Args:
            pdf_path: Path to PDF file
            lang: Language for OCR
            
        Returns:
            List of Document objects with OCR-extracted text
        """
        documents = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get page as image
                mat = fitz.Matrix(2, 2)  # Scaling factor for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(tempfile.SpooledTemporaryFile(max_size=10000000))
                image = Image.open(tempfile.BytesIO(img_data))
                
                # Extract text using OCR
                text = self.extract_text_from_image(image, lang)
                
                if text.strip():
                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            'source': str(pdf_path),
                            'page': page_num + 1,
                            'extraction_method': 'ocr',
                            'ocr_language': lang
                        }
                    )
                    documents.append(doc_obj)
            
            doc.close()
            
        except Exception as e:
            print(f"OCR extraction error for {pdf_path}: {str(e)}")
        
        return documents
    
    def process_with_fallback_ocr(self, pdf_path: Path, 
                                  native_documents: List[Document],
                                  lang: str = 'eng') -> List[Document]:
        """
        Use OCR as fallback when native text extraction fails or produces poor results.
        
        Args:
            pdf_path: Path to PDF file
            native_documents: Documents from native text extraction
            lang: Language for OCR
            
        Returns:
            Best available documents (native or OCR)
        """
        # Check if native extraction was successful
        total_native_text = sum(len(doc.page_content.strip()) for doc in native_documents)
        
        # If native extraction produced very little text, try OCR
        if total_native_text < 500 or self.is_pdf_scanned(pdf_path):
            print(f"Using OCR for {pdf_path} (native extraction insufficient)")
            ocr_documents = self.extract_text_from_pdf_ocr(pdf_path, lang)
            
            # Return OCR results if they're better
            total_ocr_text = sum(len(doc.page_content.strip()) for doc in ocr_documents)
            if total_ocr_text > total_native_text:
                return ocr_documents
        
        return native_documents


def extract_text_with_ocr_fallback(pdf_path: Path, 
                                   native_documents: List[Document],
                                   lang: str = 'eng') -> List[Document]:
    """
    Convenience function to extract text with OCR fallback.
    
    Args:
        pdf_path: Path to PDF file
        native_documents: Documents from native extraction
        lang: OCR language
        
    Returns:
        Best available documents
    """
    if not HAS_OCR:
        print("OCR not available, using native extraction only")
        return native_documents
    
    try:
        ocr_processor = OCRProcessor()
        return ocr_processor.process_with_fallback_ocr(pdf_path, native_documents, lang)
    except Exception as e:
        print(f"OCR fallback failed: {str(e)}")
        return native_documents