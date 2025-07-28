"""
Document and query summarization module with multiple summary styles.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .llm_router import LLMRouter, LLMResponse
from .search import SearchResult


class SummaryStyle(Enum):
    """Different styles of summaries."""
    PLAIN = "plain"
    TLDR = "tldr"
    BULLETS = "bullets"
    STRUCTURED = "structured"
    EXECUTIVE = "executive"


class SummaryLength(Enum):
    """Summary length options."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


@dataclass
class SummaryResult:
    """Container for summarization results."""
    summary: str
    style: str
    length: str
    source_type: str
    token_count: int
    provider: str
    sources: List[str]
    confidence_score: Optional[float] = None


class DocumentSummarizer:
    """Handles document and search result summarization."""
    
    def __init__(self, llm_router: LLMRouter):
        """
        Initialize document summarizer.
        
        Args:
            llm_router: LLM router instance for generating summaries
        """
        self.llm_router = llm_router
        
        # Style-specific prompts
        self.style_prompts = {
            SummaryStyle.PLAIN: {
                "system": "Provide a clear, concise summary in plain language.",
                "instruction": "Summarize the following content in a clear, straightforward manner:"
            },
            SummaryStyle.TLDR: {
                "system": "Create a TL;DR (Too Long; Didn't Read) summary that captures the essence quickly.",
                "instruction": "Create a TL;DR summary of the following content:"
            },
            SummaryStyle.BULLETS: {
                "system": "Create a bullet-point summary highlighting key information.",
                "instruction": "Summarize the following content using bullet points for key information:"
            },
            SummaryStyle.STRUCTURED: {
                "system": "Create a well-structured summary with clear sections and headings.",
                "instruction": "Create a structured summary with clear sections for the following content:"
            },
            SummaryStyle.EXECUTIVE: {
                "system": "Create an executive summary suitable for business or academic contexts.",
                "instruction": "Create an executive summary of the following content:"
            }
        }
        
        # Length-specific guidelines
        self.length_guidelines = {
            SummaryLength.SHORT: "Keep the summary to 2-3 sentences or 50-100 words.",
            SummaryLength.MEDIUM: "Provide a moderate summary of 1-2 paragraphs or 100-200 words.",
            SummaryLength.LONG: "Create a comprehensive summary of 3-4 paragraphs or 200-400 words."
        }
    
    def _create_summary_prompt(self, 
                              content: str, 
                              style: SummaryStyle, 
                              length: SummaryLength,
                              context: str = "") -> str:
        """Create a prompt for summarization based on style and length."""
        
        style_info = self.style_prompts[style]
        length_guideline = self.length_guidelines[length]
        
        prompt_parts = [
            f"TASK: {style_info['system']}",
            f"LENGTH: {length_guideline}",
            "",
            style_info['instruction'],
            ""
        ]
        
        if context:
            prompt_parts.extend([
                "CONTEXT:",
                context,
                ""
            ])
        
        prompt_parts.extend([
            "CONTENT TO SUMMARIZE:",
            content,
            "",
            "SUMMARY:"
        ])
        
        return "\n".join(prompt_parts)
    
    async def summarize_document(self, 
                               document: Any, 
                               style: SummaryStyle = SummaryStyle.PLAIN,
                               length: SummaryLength = SummaryLength.MEDIUM,
                               use_async: bool = True) -> SummaryResult:
        """
        Summarize a single document.
        
        Args:
            document: LangChain Document object
            style: Summary style
            length: Summary length
            use_async: Whether to use async generation
            
        Returns:
            SummaryResult object
        """
        content = document.page_content
        source = document.metadata.get('source', 'Unknown')
        
        # Create context from metadata
        context_parts = []
        if 'source' in document.metadata:
            context_parts.append(f"Source: {Path(source).name}")
        if 'file_type' in document.metadata:
            context_parts.append(f"Type: {document.metadata['file_type']}")
        if 'page' in document.metadata:
            context_parts.append(f"Page: {document.metadata['page']}")
        
        context = " | ".join(context_parts) if context_parts else ""
        
        # Create prompt
        prompt = self._create_summary_prompt(content, style, length, context)
        
        # Generate summary
        if use_async:
            response = await self.llm_router.generate_async(prompt)
        else:
            response = self.llm_router.generate(prompt)
        
        return SummaryResult(
            summary=response.output.strip(),
            style=style.value,
            length=length.value,
            source_type="document",
            token_count=response.tokens,
            provider=response.provider,
            sources=[source]
        )
    
    async def summarize_search_results(self, 
                                     search_results: List[SearchResult],
                                     query: str,
                                     style: SummaryStyle = SummaryStyle.STRUCTURED,
                                     length: SummaryLength = SummaryLength.MEDIUM,
                                     max_results: int = 5,
                                     use_async: bool = True) -> SummaryResult:
        """
        Summarize search results in context of a query.
        
        Args:
            search_results: List of search results
            query: Original search query
            style: Summary style
            length: Summary length
            max_results: Maximum number of results to include
            use_async: Whether to use async generation
            
        Returns:
            SummaryResult object
        """
        if not search_results:
            return SummaryResult(
                summary="No search results to summarize.",
                style=style.value,
                length=length.value,
                source_type="search_results",
                token_count=0,
                provider="none",
                sources=[]
            )
        
        # Prepare content from search results
        content_parts = []
        sources = []
        
        for i, result in enumerate(search_results[:max_results], 1):
            source = result.document.metadata.get('source', f'Document {i}')
            source_name = Path(source).name if source != f'Document {i}' else source
            sources.append(source)
            
            content = result.document.page_content.strip()
            if len(content) > 500:  # Limit content length
                content = content[:500] + "..."
            
            content_parts.append(f"Result {i} (from {source_name}):\n{content}")
        
        combined_content = "\n\n".join(content_parts)
        
        # Create context with query information
        context = f"These are search results for the query: '{query}'"
        
        # Create specialized prompt for search results
        if style == SummaryStyle.STRUCTURED:
            instruction = f"Create a structured summary that addresses the query '{query}' based on the search results below. Organize information by relevance and cite sources."
        elif style == SummaryStyle.EXECUTIVE:
            instruction = f"Create an executive summary addressing the query '{query}' based on the search results. Focus on key findings and conclusions."
        else:
            style_info = self.style_prompts[style]
            instruction = f"{style_info['instruction']} Focus on how the content relates to the query '{query}'."
        
        length_guideline = self.length_guidelines[length]
        
        prompt = f"""TASK: Summarize search results to answer a specific query.
LENGTH: {length_guideline}

{instruction}

QUERY: {query}
{context}

SEARCH RESULTS:
{combined_content}

SUMMARY:"""
        
        # Generate summary
        if use_async:
            response = await self.llm_router.generate_async(prompt)
        else:
            response = self.llm_router.generate(prompt)
        
        return SummaryResult(
            summary=response.output.strip(),
            style=style.value,
            length=length.value,
            source_type="search_results",
            token_count=response.tokens,
            provider=response.provider,
            sources=list(set(sources))  # Remove duplicates
        )
    
    async def summarize_multiple_documents(self, 
                                         documents: List[Any],
                                         topic: str = "",
                                         style: SummaryStyle = SummaryStyle.STRUCTURED,
                                         length: SummaryLength = SummaryLength.LONG,
                                         use_async: bool = True) -> SummaryResult:
        """
        Summarize multiple documents together.
        
        Args:
            documents: List of LangChain Document objects
            topic: Optional topic to focus the summary on
            style: Summary style
            length: Summary length
            use_async: Whether to use async generation
            
        Returns:
            SummaryResult object
        """
        if not documents:
            return SummaryResult(
                summary="No documents to summarize.",
                style=style.value,
                length=length.value,
                source_type="multiple_documents",
                token_count=0,
                provider="none",
                sources=[]
            )
        
        # Prepare content from multiple documents
        content_parts = []
        sources = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', f'Document {i}')
            source_name = Path(source).name if source != f'Document {i}' else source
            sources.append(source)
            
            content = doc.page_content.strip()
            if len(content) > 1000:  # Limit content length for multiple docs
                content = content[:1000] + "..."
            
            content_parts.append(f"Document {i} ({source_name}):\n{content}")
        
        combined_content = "\n\n".join(content_parts)
        
        # Create context
        context_parts = [f"Summary of {len(documents)} documents"]
        if topic:
            context_parts.append(f"focused on the topic: '{topic}'")
        context = " ".join(context_parts)
        
        # Create prompt
        if topic:
            instruction = f"Summarize the following documents with focus on the topic '{topic}'. Identify common themes, differences, and key insights across all documents."
        else:
            instruction = "Summarize the following documents. Identify common themes, key points, and insights across all documents."
        
        length_guideline = self.length_guidelines[length]
        
        prompt = f"""TASK: Multi-document summarization.
LENGTH: {length_guideline}

{instruction}

CONTEXT: {context}

DOCUMENTS:
{combined_content}

SUMMARY:"""
        
        # Generate summary
        if use_async:
            response = await self.llm_router.generate_async(prompt)
        else:
            response = self.llm_router.generate(prompt)
        
        return SummaryResult(
            summary=response.output.strip(),
            style=style.value,
            length=length.value,
            source_type="multiple_documents",
            token_count=response.tokens,
            provider=response.provider,
            sources=list(set(sources))
        )
    
    def batch_summarize_documents(self, 
                                documents: List[Any],
                                style: SummaryStyle = SummaryStyle.PLAIN,
                                length: SummaryLength = SummaryLength.SHORT) -> List[SummaryResult]:
        """
        Synchronously summarize multiple documents individually.
        
        Args:
            documents: List of LangChain Document objects
            style: Summary style
            length: Summary length
            
        Returns:
            List of SummaryResult objects
        """
        summaries = []
        
        for doc in documents:
            try:
                # Use synchronous version
                summary = asyncio.run(self.summarize_document(doc, style, length, use_async=False))
                summaries.append(summary)
            except Exception as e:
                # Create error summary
                source = doc.metadata.get('source', 'Unknown')
                error_summary = SummaryResult(
                    summary=f"Error summarizing document: {str(e)}",
                    style=style.value,
                    length=length.value,
                    source_type="document",
                    token_count=0,
                    provider="error",
                    sources=[source]
                )
                summaries.append(error_summary)
        
        return summaries
    
    def get_summary_styles(self) -> Dict[str, str]:
        """Get available summary styles and their descriptions."""
        return {
            "plain": "Clear, straightforward summary in natural language",
            "tldr": "Quick TL;DR style summary highlighting the essence",
            "bullets": "Bullet-point format for easy scanning",
            "structured": "Well-organized summary with clear sections",
            "executive": "Professional executive summary format"
        }
    
    def get_summary_lengths(self) -> Dict[str, str]:
        """Get available summary lengths and their descriptions."""
        return {
            "short": "Brief summary (50-100 words)",
            "medium": "Moderate summary (100-200 words)",
            "long": "Comprehensive summary (200-400 words)"
        }


class SummaryManager:
    """High-level manager for different types of summarization tasks."""
    
    def __init__(self, llm_router: LLMRouter):
        """Initialize summary manager."""
        self.summarizer = DocumentSummarizer(llm_router)
    
    async def smart_summarize(self, 
                            content: Union[Any, List[Any], List[SearchResult]],
                            query: str = "",
                            auto_style: bool = True,
                            auto_length: bool = True) -> SummaryResult:
        """
        Intelligently choose summary style and length based on content type and query.
        
        Args:
            content: Document, list of documents, or search results
            query: Optional query for context
            auto_style: Whether to automatically choose style
            auto_length: Whether to automatically choose length
            
        Returns:
            SummaryResult object
        """
        # Determine content type and best approach
        if isinstance(content, list):
            if len(content) == 0:
                style = SummaryStyle.PLAIN
                length = SummaryLength.SHORT
            elif hasattr(content[0], 'combined_score'):  # SearchResult objects
                style = SummaryStyle.STRUCTURED if auto_style else SummaryStyle.PLAIN
                length = SummaryLength.MEDIUM if auto_length else SummaryLength.MEDIUM
                return await self.summarizer.summarize_search_results(
                    content, query, style, length
                )
            else:  # List of documents
                if len(content) == 1:
                    style = SummaryStyle.PLAIN if auto_style else SummaryStyle.PLAIN
                    length = SummaryLength.MEDIUM if auto_length else SummaryLength.MEDIUM
                    return await self.summarizer.summarize_document(content[0], style, length)
                else:
                    style = SummaryStyle.STRUCTURED if auto_style else SummaryStyle.STRUCTURED
                    length = SummaryLength.LONG if auto_length else SummaryLength.LONG
                    return await self.summarizer.summarize_multiple_documents(
                        content, query, style, length
                    )
        else:
            # Single document
            style = SummaryStyle.PLAIN if auto_style else SummaryStyle.PLAIN
            length = SummaryLength.MEDIUM if auto_length else SummaryLength.MEDIUM
            return await self.summarizer.summarize_document(content, style, length)
    
    def get_available_options(self) -> Dict[str, Any]:
        """Get all available summarization options."""
        return {
            "styles": self.summarizer.get_summary_styles(),
            "lengths": self.summarizer.get_summary_lengths(),
            "supported_content_types": [
                "single_document",
                "multiple_documents", 
                "search_results"
            ]
        }