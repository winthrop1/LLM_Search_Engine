"""
Conversational AI module with multi-turn memory and context management.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .llm_router import LLMRouter, LLMResponse
from .search import HybridSearcher, SearchResult


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    timestamp: str
    user_query: str
    search_results: List[Dict[str, Any]]
    ai_response: str
    context_used: str
    tokens_used: int
    provider: str


@dataclass
class ConversationMemory:
    """Stores conversation history and context."""
    conversation_id: str
    turns: List[ConversationTurn]
    created_at: str
    last_updated: str
    total_tokens: int = 0


class ConversationManager:
    """Manages conversational interactions with memory and context."""
    
    def __init__(self, 
                 llm_router: LLMRouter,
                 hybrid_searcher: HybridSearcher,
                 memory_file: str = "./processed/conversation_memory.json",
                 max_context_turns: int = 5,
                 max_context_length: int = 4000):
        """
        Initialize conversation manager.
        
        Args:
            llm_router: LLM router instance for generating responses
            hybrid_searcher: Hybrid searcher for document retrieval
            memory_file: Path to save conversation memory
            max_context_turns: Maximum number of previous turns to include in context
            max_context_length: Maximum length of context in characters
        """
        self.llm_router = llm_router
        self.hybrid_searcher = hybrid_searcher
        self.memory_file = Path(memory_file)
        self.max_context_turns = max_context_turns
        self.max_context_length = max_context_length
        
        # Current conversation
        self.current_conversation: Optional[ConversationMemory] = None
        
        # Create memory directory if it doesn't exist
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing conversations
        self.conversations: Dict[str, ConversationMemory] = self._load_conversations()
    
    def _load_conversations(self) -> Dict[str, ConversationMemory]:
        """Load conversations from memory file."""
        if not self.memory_file.exists():
            return {}
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            conversations = {}
            for conv_id, conv_data in data.items():
                # Convert turns data back to ConversationTurn objects
                turns = []
                for turn_data in conv_data['turns']:
                    turn = ConversationTurn(**turn_data)
                    turns.append(turn)
                
                conv_data['turns'] = turns
                conversations[conv_id] = ConversationMemory(**conv_data)
            
            return conversations
        except Exception as e:
            print(f"Error loading conversations: {e}")
            return {}
    
    def _save_conversations(self):
        """Save conversations to memory file."""
        try:
            # Convert conversations to serializable format
            data = {}
            for conv_id, conv in self.conversations.items():
                conv_dict = asdict(conv)
                data[conv_id] = conv_dict
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving conversations: {e}")
    
    def start_new_conversation(self) -> str:
        """Start a new conversation and return its ID."""
        timestamp = datetime.now().isoformat()
        conv_id = f"conv_{timestamp.replace(':', '-').replace('.', '-')}"
        
        self.current_conversation = ConversationMemory(
            conversation_id=conv_id,
            turns=[],
            created_at=timestamp,
            last_updated=timestamp
        )
        
        self.conversations[conv_id] = self.current_conversation
        print(f"Started new conversation: {conv_id}")
        return conv_id
    
    def load_conversation(self, conversation_id: str) -> bool:
        """Load an existing conversation."""
        if conversation_id in self.conversations:
            self.current_conversation = self.conversations[conversation_id]
            print(f"Loaded conversation: {conversation_id}")
            return True
        else:
            print(f"Conversation not found: {conversation_id}")
            return False
    
    def _build_context(self, current_query: str) -> str:
        """Build context from previous conversation turns."""
        if not self.current_conversation or not self.current_conversation.turns:
            return ""
        
        # Get recent turns for context
        recent_turns = self.current_conversation.turns[-self.max_context_turns:]
        
        context_parts = []
        total_length = 0
        
        # Build context from most recent to oldest
        for turn in reversed(recent_turns):
            turn_context = f"User: {turn.user_query}\nAssistant: {turn.ai_response}\n\n"
            
            if total_length + len(turn_context) > self.max_context_length:
                break
            
            context_parts.insert(0, turn_context)
            total_length += len(turn_context)
        
        if context_parts:
            return "Previous conversation:\n" + "".join(context_parts) + f"Current question: {current_query}"
        else:
            return current_query
    
    def _extract_search_context(self, search_results: List[SearchResult]) -> str:
        """Extract relevant context from search results."""
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        total_length = 0
        max_context = 2000  # Limit context length
        
        for i, result in enumerate(search_results[:5]):  # Use top 5 results
            # Create context snippet
            content = result.document.page_content.strip()
            if len(content) > 300:
                content = content[:300] + "..."
            
            source = result.document.metadata.get('source', 'Unknown')
            source_name = Path(source).name if source != 'Unknown' else 'Unknown'
            
            snippet = f"Document {i+1} ({source_name}):\n{content}\n"
            
            if total_length + len(snippet) > max_context:
                break
            
            context_parts.append(snippet)
            total_length += len(snippet)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, search_context: str, conversation_context: str) -> str:
        """Create a comprehensive prompt for the LLM."""
        prompt_parts = [
            "You are an AI assistant helping users find information from a document collection.",
            "Use the provided documents and conversation history to answer questions accurately.",
            "",
            "INSTRUCTIONS:",
            "- Answer based on the provided documents when possible",
            "- Reference specific documents when citing information",
            "- If information isn't in the documents, say so clearly",
            "- Consider the conversation history for context",
            "- Be concise but thorough in your responses",
            "",
        ]
        
        if conversation_context and conversation_context != query:
            prompt_parts.extend([
                "CONVERSATION HISTORY:",
                conversation_context,
                "",
            ])
        
        prompt_parts.extend([
            "RELEVANT DOCUMENTS:",
            search_context,
            "",
            "QUESTION:",
            query,
            "",
            "Please provide a helpful answer based on the documents and conversation context."
        ])
        
        return "\n".join(prompt_parts)
    
    async def ask_question(self, 
                          query: str, 
                          search_top_k: int = 5,
                          use_streaming: bool = None) -> Tuple[str, List[SearchResult]]:
        """
        Ask a question and get a response with context from documents and conversation history.
        
        Args:
            query: User's question
            search_top_k: Number of documents to retrieve
            use_streaming: Whether to use streaming response
            
        Returns:
            Tuple of (response, search_results)
        """
        if not self.current_conversation:
            self.start_new_conversation()
        
        # Use streaming preference from LLM router if not explicitly specified
        if use_streaming is None:
            use_streaming = self.llm_router.use_streaming
        
        # Perform hybrid search
        print(f"Searching for relevant documents...")
        search_results = self.hybrid_searcher.search(query, top_k=search_top_k)
        
        # Build conversation context
        conversation_context = self._build_context(query)
        
        # Extract search context
        search_context = self._extract_search_context(search_results)
        
        # Create comprehensive prompt
        prompt = self._create_prompt(query, search_context, conversation_context)
        
        # Generate response
        try:
            if use_streaming:
                print("🌊 Generating response (streaming)...")
                print("💭 ", end='', flush=True)  # Thinking indicator
                response_parts = []
                chunk_count = 0
                
                for chunk in self.llm_router.stream(prompt):
                    print(chunk, end='', flush=True)
                    response_parts.append(chunk)
                    chunk_count += 1
                    
                    # Add periodic indicators for very long responses
                    if chunk_count % 50 == 0:
                        print(" ", end='', flush=True)
                
                print()  # New line after streaming
                print("✅ Streaming complete")
                
                response = ''.join(response_parts)
                # Estimate token count for streaming
                tokens_used = len(prompt.split()) + len(response.split())
                provider = self.llm_router.provider_name
            else:
                print("🤖 Generating response...")
                llm_response = self.llm_router.generate(prompt)
                response = llm_response.output
                tokens_used = llm_response.tokens
                provider = llm_response.provider
        
        except Exception as e:
            response = f"Error generating response: {e}"
            tokens_used = 0
            provider = "error"
        
        # Save conversation turn
        turn = ConversationTurn(
            timestamp=datetime.now().isoformat(),
            user_query=query,
            search_results=[{
                'source': result.document.metadata.get('source', 'Unknown'),
                'semantic_score': result.semantic_score,
                'keyword_score': result.keyword_score,
                'combined_score': result.combined_score,
                'content_preview': result.document.page_content[:200] + "..." if len(result.document.page_content) > 200 else result.document.page_content
            } for result in search_results],
            ai_response=response,
            context_used=conversation_context,
            tokens_used=tokens_used,
            provider=provider
        )
        
        self.current_conversation.turns.append(turn)
        self.current_conversation.last_updated = datetime.now().isoformat()
        self.current_conversation.total_tokens += tokens_used
        
        # Save to disk
        self._save_conversations()
        
        return response, search_results
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        if not self.current_conversation:
            return {"error": "No active conversation"}
        
        return {
            "conversation_id": self.current_conversation.conversation_id,
            "created_at": self.current_conversation.created_at,
            "last_updated": self.current_conversation.last_updated,
            "total_turns": len(self.current_conversation.turns),
            "total_tokens": self.current_conversation.total_tokens,
            "provider": self.llm_router.provider_name,
            "model": self.llm_router.model
        }
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all saved conversations."""
        return [
            {
                "conversation_id": conv.conversation_id,
                "created_at": conv.created_at,
                "last_updated": conv.last_updated,
                "total_turns": len(conv.turns),
                "total_tokens": conv.total_tokens
            }
            for conv in self.conversations.values()
        ]
    
    def export_conversation(self, conversation_id: str = None) -> Dict[str, Any]:
        """Export conversation for analysis or backup."""
        if conversation_id:
            if conversation_id not in self.conversations:
                return {"error": f"Conversation {conversation_id} not found"}
            conv = self.conversations[conversation_id]
        else:
            if not self.current_conversation:
                return {"error": "No active conversation"}
            conv = self.current_conversation
        
        return asdict(conv)
    
    def clear_conversation_history(self, conversation_id: str = None):
        """Clear conversation history."""
        if conversation_id:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                if self.current_conversation and self.current_conversation.conversation_id == conversation_id:
                    self.current_conversation = None
        else:
            # Clear all conversations
            self.conversations.clear()
            self.current_conversation = None
        
        self._save_conversations()
        print("Conversation history cleared")


class ConversationAnalyzer:
    """Analyze conversation patterns and provide insights."""
    
    @staticmethod
    def analyze_conversation(conversation: ConversationMemory) -> Dict[str, Any]:
        """Analyze a conversation for patterns and insights."""
        if not conversation.turns:
            return {"error": "No turns to analyze"}
        
        # Basic statistics
        total_turns = len(conversation.turns)
        total_tokens = sum(turn.tokens_used for turn in conversation.turns)
        avg_tokens_per_turn = total_tokens / total_turns if total_turns > 0 else 0
        
        # Query analysis
        query_lengths = [len(turn.user_query) for turn in conversation.turns]
        avg_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0
        
        # Search effectiveness
        search_results_counts = [len(turn.search_results) for turn in conversation.turns]
        avg_results_per_query = sum(search_results_counts) / len(search_results_counts) if search_results_counts else 0
        
        # Provider usage
        providers = [turn.provider for turn in conversation.turns]
        provider_counts = {provider: providers.count(provider) for provider in set(providers)}
        
        # Time analysis
        timestamps = [turn.timestamp for turn in conversation.turns]
        conversation_duration = None
        if len(timestamps) > 1:
            from datetime import datetime
            start_time = datetime.fromisoformat(timestamps[0])
            end_time = datetime.fromisoformat(timestamps[-1])
            conversation_duration = str(end_time - start_time)
        
        return {
            "conversation_id": conversation.conversation_id,
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "avg_tokens_per_turn": round(avg_tokens_per_turn, 2),
            "avg_query_length": round(avg_query_length, 2),
            "avg_results_per_query": round(avg_results_per_query, 2),
            "provider_usage": provider_counts,
            "conversation_duration": conversation_duration,
            "created_at": conversation.created_at,
            "last_updated": conversation.last_updated
        }