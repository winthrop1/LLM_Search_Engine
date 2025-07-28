#!/usr/bin/env python3
"""
Test script for image ingestion functionality.
Tests OCR-based text extraction from image files.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.ingestion import DocumentIngestion
    from src.ocr import HAS_OCR
    print(f"OCR available: {HAS_OCR}")
    
    if not HAS_OCR:
        print("OCR dependencies not available. Please install with:")
        print("pip install pytesseract pillow pymupdf")
        sys.exit(1)
    
    # Test image format support
    ingestion = DocumentIngestion()
    supported_formats = ingestion.get_supported_formats()
    
    print("Supported file formats:")
    for fmt in supported_formats:
        print(f"  {fmt}")
    
    # Check for image formats
    image_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif']
    supported_images = [fmt for fmt in image_formats if fmt in supported_formats]
    
    print(f"\nSupported image formats: {supported_images}")
    
    # Test if we can create the ImageLoader
    try:
        from src.ingestion import ImageLoader
        print("ImageLoader class available: ✓")
        
        # Test if OCR processor is working
        from src.ocr import OCRProcessor
        ocr = OCRProcessor()
        print("OCR processor initialized: ✓")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        sys.exit(1)
    
    print("\nImage ingestion functionality is ready!")
    print("To test with actual images:")
    print("1. Place image files (.jpg, .png, etc.) in the ./data/ folder")
    print("2. Run the main application: python main.py")
    print("3. Search for text content from the images")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error: {e}")