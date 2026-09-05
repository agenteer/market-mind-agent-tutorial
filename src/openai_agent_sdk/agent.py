# src/openai_agent_sdk/agent.py
from agents import Agent, Runner, RunContextWrapper, function_tool
import logging
from dataclasses import dataclass
from typing import Optional
from functools import wraps
from src.openai_agent_sdk.memory import ConversationMemory
from src.common.config import SYSTEM_PROMPT, DEFAULT_MODEL
from src.common.tools_yf import (
    get_stock_price as original_get_stock_price,
    get_stock_history as original_get_stock_history,
    get_company_info as original_get_company_info,
    get_financial_metrics as original_get_financial_metrics
)

# Configure logger for the agent
logger = logging.getLogger(__name__)

# Wrap our tools to use with the Agent SDK
@function_tool
@wraps(original_get_stock_price)
def get_stock_price(ticker: str) -> str:
    return original_get_stock_price(ticker)

@function_tool
@wraps(original_get_stock_history)
def get_stock_history(ticker: str, days: int) -> str:
    return original_get_stock_history(ticker, days)

@function_tool
@wraps(original_get_company_info)
def get_company_info(ticker: str) -> str:
    return original_get_company_info(ticker)

@function_tool
@wraps(original_get_financial_metrics)
def get_financial_metrics(ticker: str) -> str:
    return original_get_financial_metrics(ticker)

@dataclass
class MarketMindContext:
    """Context object for the MarketMind agent."""
    memory: ConversationMemory
    previous_response_id: Optional[str] = None
    use_explicit_memory: bool = True
    use_response_id_memory: bool = True

    def add_to_memory(self, user_query: str, agent_response: str) -> None:
        """Add an interaction to the conversation memory."""
        if self.use_explicit_memory:
            self.memory.add_interaction(user_query, agent_response)

    def get_memory_summary(self) -> str:
        """Get a summary of the conversation memory."""
        if self.use_explicit_memory:
            return self.memory.get_conversation_summary()
        return ""  # Return empty string if explicit memory is disabled

    def set_response_id(self, response_id: Optional[str]) -> None:
        """Store the response ID from the last interaction."""
        if self.use_response_id_memory and response_id:
            logger.debug(f"Storing response ID: {response_id}")
            self.previous_response_id = response_id
        else:
            logger.debug("No response ID to store or response ID memory disabled")
            self.previous_response_id = None

class MarketMindOpenAIAgent:
    def __init__(self,
                 model=DEFAULT_MODEL,
                 use_explicit_memory=True,
                 use_response_id_memory=True):
        """Initialize the MarketMind agent.

        Args:
            model: The OpenAI model to use.
            use_explicit_memory: Whether to use the explicit conversation memory.
            use_response_id_memory: Whether to use the response ID for conversation continuity.
        """
        logger.info(f"Initializing MarketMindOpenAIAgent with model={model}, " +
                   f"use_explicit_memory={use_explicit_memory}, " +
                   f"use_response_id_memory={use_response_id_memory}")

        # Initialize context with memory
        self.context = MarketMindContext(
            memory=ConversationMemory(),
            use_explicit_memory=use_explicit_memory,
            use_response_id_memory=use_response_id_memory
        )

        # Initialize the agent with proper typing
        self.agent = Agent[MarketMindContext](
            name = "MarketMind",
            model = model,
            instructions = self._get_dynamic_instructions,
            tools = [
                get_stock_price,
                get_stock_history,
                get_company_info,
                get_financial_metrics
            ],
        )
        logger.info("Agent initialization complete")

    def _get_dynamic_instructions(self, context: RunContextWrapper[MarketMindContext], agent=None) -> str:
        """Dynamic instructions that include conversation memory."""
        logger.debug("Generating dynamic instructions with conversation memory")

        # Get the actual context object
        market_mind_context = context.context

        if market_mind_context.use_explicit_memory:
            logger.debug(f"Context received in instructions: memory_size={len(market_mind_context.memory.conversation_history) if market_mind_context.memory.conversation_history else 0}")
        else:
            logger.debug("Explicit memory disabled")

        base_instructions = SYSTEM_PROMPT

        # Add memory context if conversation history exists and explicit memory is enabled
        if (market_mind_context.use_explicit_memory and
            market_mind_context.memory and
            market_mind_context.memory.conversation_history):
            memory_context = f"\n\nCONVERSATION MEMORY:\n{market_mind_context.get_memory_summary()}"
            full_instructions = base_instructions + memory_context
            logger.debug(f"Added conversation memory context ({len(market_mind_context.memory.conversation_history)} interactions)")
            return full_instructions

        logger.debug("No conversation memory to add or explicit memory disabled")
        return base_instructions

    async def process_query(self, query: str, *,
                     max_turns: int = 10,
                     hooks = None,
                     run_config = None,
                     previous_response_id: str = None) -> str:
        """Process a user query using the agent and update memory.

        Args:
            query: The user's query string
            max_turns: The maximum number of turns to run the agent for
            hooks: An object that receives callbacks on various lifecycle events
            run_config: Global settings for the entire agent run
            previous_response_id: The ID of the previous response, if using OpenAI models via the
                Responses API, this allows you to skip passing in input from the previous turn.
        """
        logger.info(f"Processing query: {query[:50]}...")

        # Only use stored response_id if explicitly enabled and not provided
        if (self.context.use_response_id_memory and
            previous_response_id is None and
            self.context.previous_response_id):
            logger.debug(f"Using stored response ID: {self.context.previous_response_id}")
            previous_response_id = self.context.previous_response_id
        elif not self.context.use_response_id_memory:
            # If response ID memory is disabled, explicitly set to None
            previous_response_id = None
            logger.debug("Response ID memory disabled, ignoring any previous response ID")

        # Process the query using the agent with proper context
        logger.debug(f"Sending query to OpenAI agent with context type: {type(self.context)}")

        # Pass all the parameters to the Runner.run method
        result = await Runner.run(
            self.agent,
            query,
            context=self.context,  # Pass the context to the runner
            max_turns=max_turns,
            hooks=hooks,
            run_config=run_config,
            previous_response_id=previous_response_id
        )

        # Log usage information if available
        if hasattr(result, 'usage'):
            logger.debug(f"Usage stats: {result.usage}")

        # Store the latest response_id for future interactions if feature is enabled
        if self.context.use_response_id_memory:
            latest_response_id = None
            if result.raw_responses and hasattr(result.raw_responses[-1], 'response_id'):
                latest_response_id = result.raw_responses[-1].response_id
                logger.debug(f"Got response_id: {latest_response_id}")

            # Update the stored response_id in the context
            self.context.set_response_id(latest_response_id)
        else:
            logger.debug("Response ID tracking disabled, not storing response ID")

        final_output = result.final_output
        logger.debug(f"Received response: {final_output[:50]}...")

        # Store the conversation history using the context if explicit memory is enabled
        logger.info("Updating conversation memory")
        self.context.add_to_memory(query, final_output)

        return final_output
