"""
Hybrid search module combining semantic and keyword search capabilities.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np


@dataclass
class SearchResult:
    """Container for search results with scoring information."""
    document: Any  # LangChain Document object
    semantic_score: float
    keyword_score: float
    combined_score: float
    match_highlights: List[str]


class KeywordSearcher:
    """Handles keyword-based search with fuzzy matching and TF-IDF scoring."""
    
    def __init__(self, documents: List[Any]):
        """
        Initialize keyword searcher.
        
        Args:
            documents: List of LangChain Document objects
        """
        self.documents = documents
        self.doc_texts = [doc.page_content.lower() for doc in documents]
        self.word_freq = self._build_word_frequency()
        self.doc_freq = self._build_document_frequency()
        self.total_docs = len(documents)
    
    def _build_word_frequency(self) -> List[Dict[str, int]]:
        """Build word frequency for each document."""
        word_freq = []
        for text in self.doc_texts:
            # Simple tokenization - split on non-alphanumeric characters
            words = re.findall(r'\b\w+\b', text.lower())
            freq = defaultdict(int)
            for word in words:
                freq[word] += 1
            word_freq.append(dict(freq))
        return word_freq
    
    def _build_document_frequency(self) -> Dict[str, int]:
        """Build document frequency for all words."""
        doc_freq = defaultdict(int)
        for word_freq in self.word_freq:
            for word in word_freq:
                doc_freq[word] += 1
        return dict(doc_freq)
    
    def _calculate_tfidf(self, word: str, doc_idx: int) -> float:
        """Calculate TF-IDF score for a word in a document."""
        if word not in self.word_freq[doc_idx]:
            return 0.0
        
        # Term frequency
        tf = self.word_freq[doc_idx][word]
        
        # Inverse document frequency
        df = self.doc_freq.get(word, 0)
        if df == 0:
            return 0.0
        
        idf = math.log(self.total_docs / df)
        
        return tf * idf
    
    def _fuzzy_match(self, query_word: str, text: str, threshold: float = 0.8) -> List[str]:
        """Find fuzzy matches for a query word in text."""
        words = re.findall(r'\b\w+\b', text.lower())
        matches = []
        
        for word in words:
            # Simple character-based similarity
            if query_word == word:
                matches.append(word)
            elif len(query_word) > 3 and len(word) > 3:
                # Jaccard similarity for longer words
                set1 = set(query_word)
                set2 = set(word)
                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                if similarity >= threshold:
                    matches.append(word)
        
        return matches
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float, List[str]]]:
        """
        Perform keyword search.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_index, score, highlighted_matches)
        """
        query_words = re.findall(r'\b\w+\b', query.lower())
        if not query_words:
            return []
        
        doc_scores = []
        
        for doc_idx, (doc, text) in enumerate(zip(self.documents, self.doc_texts)):
            score = 0.0
            matches = []
            
            for query_word in query_words:
                # Exact match TF-IDF
                tfidf_score = self._calculate_tfidf(query_word, doc_idx)
                
                # Fuzzy matches
                fuzzy_matches = self._fuzzy_match(query_word, text)
                
                if tfidf_score > 0:
                    score += tfidf_score
                    matches.append(query_word)
                elif fuzzy_matches:
                    # Lower score for fuzzy matches
                    for match in fuzzy_matches:
                        fuzzy_tfidf = self._calculate_tfidf(match, doc_idx)
                        score += fuzzy_tfidf * 0.5  # Penalty for fuzzy match
                        matches.append(match)
            
            if score > 0:
                doc_scores.append((doc_idx, score, matches))
        
        # Sort by score and return top_k
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]


class HybridSearcher:
    """Combines semantic and keyword search with configurable weighting."""
    
    def __init__(self, 
                 embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 semantic_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """
        Initialize hybrid searcher.
        
        Args:
            embeddings_model: Name of the sentence transformer model
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """
        self.embeddings_model = SentenceTransformer(embeddings_model)
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
        # Normalize weights
        total_weight = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight
        
        self.documents = None
        self.keyword_searcher = None
        self.document_embeddings = None
    
    def index_documents(self, documents: List[Any]):
        """
        Index documents for hybrid search.
        
        Args:
            documents: List of LangChain Document objects
        """
        self.documents = documents
        
        # Initialize keyword searcher
        self.keyword_searcher = KeywordSearcher(documents)
        
        # Create embeddings for semantic search
        texts = [doc.page_content for doc in documents]
        print(f"Creating embeddings for {len(texts)} documents...")
        self.document_embeddings = self.embeddings_model.encode(texts)
        print("Embeddings created successfully")
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Perform semantic search using sentence transformers."""
        if self.document_embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.embeddings_model.encode([query])
        
        # Calculate cosine similarity
        similarities = np.dot(query_embedding, self.document_embeddings.T)[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               min_score_threshold: float = 0.0) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword approaches.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score_threshold: Minimum combined score threshold
            
        Returns:
            List of SearchResult objects
        """
        if not self.documents:
            return []
        
        # Perform semantic search
        semantic_results = self._semantic_search(query, top_k * 2)  # Get more for better combination
        
        # Perform keyword search
        keyword_results = self.keyword_searcher.search(query, top_k * 2)
        
        # Combine results
        combined_scores = {}
        highlights = {}
        
        # Normalize semantic scores (0-1 range)
        if semantic_results:
            max_semantic = max(score for _, score in semantic_results)
            min_semantic = min(score for _, score in semantic_results)
            semantic_range = max_semantic - min_semantic if max_semantic > min_semantic else 1
            
            for doc_idx, score in semantic_results:
                normalized_score = (score - min_semantic) / semantic_range
                combined_scores[doc_idx] = {
                    'semantic': normalized_score,
                    'keyword': 0.0
                }
                highlights[doc_idx] = []
        
        # Normalize keyword scores (0-1 range)
        if keyword_results:
            max_keyword = max(score for _, score, _ in keyword_results)
            min_keyword = min(score for _, score, _ in keyword_results)
            keyword_range = max_keyword - min_keyword if max_keyword > min_keyword else 1
            
            for doc_idx, score, matches in keyword_results:
                normalized_score = (score - min_keyword) / keyword_range
                
                if doc_idx in combined_scores:
                    combined_scores[doc_idx]['keyword'] = normalized_score
                else:
                    combined_scores[doc_idx] = {
                        'semantic': 0.0,
                        'keyword': normalized_score
                    }
                highlights[doc_idx] = matches
        
        # Calculate final combined scores
        final_results = []
        for doc_idx, scores in combined_scores.items():
            semantic_score = scores['semantic']
            keyword_score = scores['keyword']
            
            combined_score = (
                self.semantic_weight * semantic_score + 
                self.keyword_weight * keyword_score
            )
            
            if combined_score >= min_score_threshold:
                result = SearchResult(
                    document=self.documents[doc_idx],
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    combined_score=combined_score,
                    match_highlights=highlights.get(doc_idx, [])
                )
                final_results.append(result)
        
        # Sort by combined score and return top_k
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        return final_results[:top_k]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents."""
        if not self.documents:
            return {'indexed_documents': 0}
        
        return {
            'indexed_documents': len(self.documents),
            'embeddings_model': self.embeddings_model.get_sentence_embedding_dimension(),
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'total_vocabulary': len(self.keyword_searcher.doc_freq) if self.keyword_searcher else 0
        }