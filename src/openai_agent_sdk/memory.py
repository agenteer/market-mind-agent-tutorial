# src/openai_agent_sdk/memory.py
from dataclasses import dataclass, field
from typing import List, Dict
import logging

# Configure logger for memory system
logger = logging.getLogger(__name__)

@dataclass
class ConversationMemory:
    """Simple memory store for conversation history."""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    max_history_size: int = 10  # Configurable parameter for memory size

    def __post_init__(self):
        """Initialize the memory system."""
        logger.info(f"Initializing ConversationMemory with max_history_size={self.max_history_size}")

    def add_interaction(self, user_query: str, agent_response: str):
        """Add a user-agent interaction to memory."""
        logger.info(f"Adding interaction to memory (history size: {len(self.conversation_history)})")
        logger.debug(f"User query: {user_query[:50]}...")

        # Just store the conversation messages
        self.conversation_history.append({
            "user_query": user_query,
            "agent_response": agent_response
        })

        # Debug log to show the full memory content
        logger.debug(f"Current memory content: {self.conversation_history}")

        # Trim history if needed
        if len(self.conversation_history) > self.max_history_size:
            logger.info(f"Trimming conversation history to max size {self.max_history_size}")
            self.conversation_history = self.conversation_history[-self.max_history_size:]

    def get_conversation_summary(self) -> str:
        """Format the conversation history for inclusion in agent context."""
        if not self.conversation_history:
            logger.debug("No conversation history available")
            return "No conversation history yet."

        logger.debug(f"Generating conversation summary for {len(self.conversation_history)} interactions")
        summary = f"Previous conversation history:\n\n"

        # Include all stored exchanges, which will be limited by max_history_size
        for interaction in self.conversation_history:
            summary += f"User: {interaction['user_query']}\n"
            summary += f"AI: {interaction['agent_response']}\n\n"

        return summary

    def clear(self):
        """Clear all conversation history."""
        logger.info("Clearing conversation history")
        self.conversation_history = []
