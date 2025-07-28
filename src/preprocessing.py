"""
Advanced text preprocessing module for document cleaning and normalization.
Handles header/footer removal, hyphenation fixes, whitespace normalization, etc.
"""

import re
from typing import List, Dict, Set
from langchain_core.documents import Document


class TextPreprocessor:
    """Advanced text preprocessing for document content."""
    
    def __init__(self):
        """Initialize text preprocessor with patterns and rules."""
        
        # Common header/footer patterns
        self.header_footer_patterns = [
            r'^\s*page\s+\d+\s*$',  # Page numbers
            r'^\s*\d+\s*$',  # Standalone numbers
            r'^\s*chapter\s+\d+.*$',  # Chapter headers
            r'^\s*\d+\.\d+.*confidential.*$',  # Confidential headers
            r'^\s*proprietary.*$',  # Proprietary notices
            r'^\s*copyright.*$',  # Copyright notices
            r'^\s*©.*$',  # Copyright symbols
            r'^\s*printed\s+on.*$',  # Print timestamps
            r'^\s*generated\s+on.*$',  # Generation timestamps
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.header_footer_patterns]
        
        # Common hyphenation patterns
        self.hyphenation_patterns = [
            (r'(\w+)-\s*\n\s*(\w+)', r'\1\2'),  # Basic hyphenation
            (r'(\w+)-\s+(\w+)', r'\1\2'),  # Hyphenation with spaces
        ]
        
        # Table detection patterns
        self.table_patterns = [
            re.compile(r'^\s*\|.*\|\s*$'),  # Markdown tables
            re.compile(r'^\s*[\w\s]+\s+[\d\.\,\s]+\s*$'),  # Data rows
        ]
        
        # Figure/table caption patterns
        self.caption_patterns = [
            re.compile(r'^(Figure|Fig\.?|Table|Chart)\s+\d+[:\.]?\s*(.+)$', re.IGNORECASE),
            re.compile(r'^(Source|Note):\s*(.+)$', re.IGNORECASE),
        ]
    
    def remove_headers_footers(self, text: str, threshold: float = 0.7) -> str:
        """
        Remove common headers and footers from text.
        
        Args:
            text: Input text
            threshold: Confidence threshold for removal
            
        Returns:
            Text with headers/footers removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Check against header/footer patterns
            is_header_footer = any(pattern.match(line_stripped) 
                                 for pattern in self.compiled_patterns)
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def fix_hyphenation(self, text: str) -> str:
        """
        Fix hyphenated words split across line breaks.
        
        Args:
            text: Input text with potential hyphenation issues
            
        Returns:
            Text with hyphenation fixed
        """
        result = text
        
        for pattern, replacement in self.hyphenation_patterns:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace and special characters.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Normalize unicode characters
        text = text.replace('\u2019', "'")  # Right single quotation mark
        text = text.replace('\u2018', "'")  # Left single quotation mark
        text = text.replace('\u201c', '"')  # Left double quotation mark
        text = text.replace('\u201d', '"')  # Right double quotation mark
        text = text.replace('\u2013', '-')  # En dash
        text = text.replace('\u2014', '--')  # Em dash
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        
        return text.strip()
    
    def extract_tables_and_captions(self, text: str) -> Dict[str, List[str]]:
        """
        Extract and identify tables and figure captions.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with 'tables', 'captions', and 'cleaned_text'
        """
        lines = text.split('\n')
        tables = []
        captions = []
        cleaned_lines = []
        
        current_table = []
        in_table = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for captions
            caption_match = None
            for pattern in self.caption_patterns:
                match = pattern.match(line_stripped)
                if match:
                    caption_match = match
                    break
            
            if caption_match:
                captions.append({
                    'type': caption_match.group(1),
                    'content': caption_match.group(2) if len(caption_match.groups()) > 1 else line_stripped
                })
                cleaned_lines.append(line)  # Keep captions in text
                continue
            
            # Check for table rows
            is_table_row = any(pattern.match(line_stripped) 
                             for pattern in self.table_patterns)
            
            if is_table_row:
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            else:
                if in_table and current_table:
                    tables.append('\n'.join(current_table))
                    current_table = []
                    in_table = False
                
                cleaned_lines.append(line)
        
        # Handle table at end of text
        if in_table and current_table:
            tables.append('\n'.join(current_table))
        
        return {
            'tables': tables,
            'captions': captions,
            'cleaned_text': '\n'.join(cleaned_lines)
        }
    
    def remove_noise_content(self, text: str) -> str:
        """
        Remove common noise content from documents.
        
        Args:
            text: Input text
            
        Returns:
            Text with noise removed
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove standalone numbers that are likely page numbers or IDs
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text
    
    def preprocess_document(self, document: Document, 
                          remove_headers: bool = True,
                          fix_hyphenation: bool = True,
                          normalize_whitespace: bool = True,
                          extract_structured_content: bool = True,
                          remove_noise: bool = True) -> Document:
        """
        Apply comprehensive preprocessing to a document.
        
        Args:
            document: Input document
            remove_headers: Remove headers and footers
            fix_hyphenation: Fix hyphenated words
            normalize_whitespace: Normalize whitespace
            extract_structured_content: Extract tables and captions
            remove_noise: Remove noise content
            
        Returns:
            Preprocessed document with updated metadata
        """
        text = document.page_content
        processing_steps = []
        
        # Track original length
        original_length = len(text)
        
        if remove_headers:
            text = self.remove_headers_footers(text)
            processing_steps.append('header_footer_removal')
        
        if fix_hyphenation:
            text = self.fix_hyphenation(text)
            processing_steps.append('hyphenation_fix')
        
        if normalize_whitespace:
            text = self.normalize_whitespace(text)
            processing_steps.append('whitespace_normalization')
        
        if extract_structured_content:
            extracted = self.extract_tables_and_captions(text)
            text = extracted['cleaned_text']
            
            # Add structured content to metadata
            if extracted['tables']:
                document.metadata['tables'] = extracted['tables']
            if extracted['captions']:
                document.metadata['captions'] = extracted['captions']
            
            processing_steps.append('structured_content_extraction')
        
        if remove_noise:
            text = self.remove_noise_content(text)
            processing_steps.append('noise_removal')
        
        # Update document
        document.page_content = text
        document.metadata.update({
            'preprocessing_steps': processing_steps,
            'original_length': original_length,
            'processed_length': len(text),
            'compression_ratio': len(text) / original_length if original_length > 0 else 0
        })
        
        return document
    
    def preprocess_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Preprocess a list of documents.
        
        Args:
            documents: List of documents to preprocess
            **kwargs: Preprocessing options
            
        Returns:
            List of preprocessed documents
        """
        processed_docs = []
        
        for doc in documents:
            try:
                processed_doc = self.preprocess_document(doc, **kwargs)
                # Only keep documents with meaningful content
                if len(processed_doc.page_content.strip()) > 50:
                    processed_docs.append(processed_doc)
            except Exception as e:
                print(f"Error preprocessing document: {str(e)}")
                # Keep original document if preprocessing fails
                processed_docs.append(doc)
        
        return processed_docs


def preprocess_documents(documents: List[Document], **kwargs) -> List[Document]:
    """
    Convenience function to preprocess documents.
    
    Args:
        documents: List of documents to preprocess
        **kwargs: Preprocessing options
        
    Returns:
        List of preprocessed documents
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_documents(documents, **kwargs)