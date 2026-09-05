# src/agent_from_scratch/agent_chat.py
import json
import logging
from openai import OpenAI
from src.agent_from_scratch.tool_manager import ToolManager
from src.common.config import DEFAULT_MODEL, SYSTEM_PROMPT, check_api_key, DEFAULT_MAX_ITERATIONS

DEFAULT_HISTORY_SIZE = 20

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

class MarketMindChatAgent:
    """
    An AI-powered financial assistant that can answer queries about stocks and markets.
    """

    def __init__(self, model=DEFAULT_MODEL):
        logger.debug(f"Initializing MarketMindChatAgent with model: {model}")
        self.tool_manager = ToolManager()
        self.model = model
        self.client = OpenAI(api_key=check_api_key())
        self.conversation_history = []

        # Initialize empty tool schemas
        self.tool_schemas = []
        logger.debug("Tool schemas initialized (empty)")

        logger.debug(f"MarketMindChatAgent initialized successfully with model: {model}")

    def register_tool(self, name, description, tool_function):
        """Register a new tool with the agent."""
        logger.debug(f"Registering tool: {name} - {description}")
        self.tool_manager.register_tool(name, description, tool_function)

        # Update tool schemas immediately
        self.tool_schemas = self.tool_manager.get_schema_for_tools()
        logger.debug(f"Tool schemas updated: now have {len(self.tool_schemas)} tools")

        logger.debug(f"Tool registered successfully: {name}")
        return self

    def _handle_tool_calls(self, message, messages):
        """
        Handle tool calls from the LLM response.

        Args:
            message: The message from the LLM containing tool calls
            messages: The conversation history to append tool results to

        Returns:
            True if tool calls were handled, False otherwise
        """
        if not message.tool_calls:
            logger.debug("No tool calls to handle")
            return False

        logger.debug(f"Processing {len(message.tool_calls)} tool calls")
        for i, tool_call in enumerate(message.tool_calls):
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            logger.debug(f"Tool call {i+1}: {function_name} with args: {function_args}")

            # Execute the tool
            logger.debug(f"Executing tool: {function_name}")
            tool_result = self.tool_manager.execute_tool(function_name, **function_args)
            logger.debug(f"Tool execution result: {tool_result}")

            # Add the tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })
            logger.debug(f"Added tool result to messages. Message count: {len(messages)}")

        return True

    def process_query(self, query):
        """
        Process a user query using the LLM model with function calling.
        Handles multiple tool calls in sequence.
        """
        logger.debug(f"Processing query: {query}")
        try:
            # Add the user query to the conversation history
            self.conversation_history.append({"role": "user", "content": query})
            logger.debug(f"Added query to conversation history. History length: {len(self.conversation_history)}")

            # System prompt that defines the agent's capabilities
            system_prompt = SYSTEM_PROMPT

            # Start with system message and conversation history
            messages = [{"role": "system", "content": system_prompt}] + self.conversation_history
            logger.debug(f"Prepared messages for API call. Message count: {len(messages)}")

            # Safety mechanism to prevent infinite loops
            iteration = 0

            # Continue the conversation until no more tool calls are needed or max iterations reached
            while iteration < DEFAULT_MAX_ITERATIONS:
                iteration += 1
                logger.debug(f"Starting iteration {iteration} of conversation loop (max: {DEFAULT_MAX_ITERATIONS})")

                # Call the model with function calling capability
                logger.debug(f"Calling API with model: {self.model}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto"
                )

                # Extract the assistant's message
                message = response.choices[0].message
                logger.debug("Received response from API")
                logger.debug(
                    f"Response details: role: {message.role}, "
                    f"content: {message.content or '[no content]'}, "
                    f"tool_calls: {message.tool_calls}"
                )

                # Add the assistant's response to messages - handle null content
                messages.append({
                    "role": message.role,
                    "content": message.content or "",  # Use empty string instead of null
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in (message.tool_calls or [])
                    ] if message.tool_calls else None
                })
                logger.debug(f"Added assistant's message to messages. Message count: {len(messages)}")

                # If there are no tool calls, we're done
                if not message.tool_calls:
                    logger.debug("No tool calls in response, finishing conversation")
                    self.conversation_history = messages[1:]  # Skip the system message
                    if len(self.conversation_history) > DEFAULT_HISTORY_SIZE:
                        logger.debug(f"Trimming conversation history from {len(self.conversation_history)} messages")
                        self.conversation_history = self.conversation_history[-DEFAULT_HISTORY_SIZE:]
                    return message.content or ""  # Return the final response

                # Handle tool calls
                self._handle_tool_calls(message, messages)

            # If we reached the maximum number of iterations, return a message about it
            logger.warning(f"Reached maximum number of iterations ({DEFAULT_MAX_ITERATIONS})")
            self.conversation_history = messages[1:]  # Skip the system message
            if len(self.conversation_history) > DEFAULT_HISTORY_SIZE:
                logger.debug(f"Trimming conversation history from {len(self.conversation_history)} messages")
                self.conversation_history = self.conversation_history[-DEFAULT_HISTORY_SIZE:]
            return (
                f"I've made multiple attempts to process your query but couldn't reach a final answer. "
                f"This might indicate a complex request or an issue with the available tools. "
                f"Please try rephrasing your question or breaking it into smaller parts."
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"An error occurred: {str(e)}"
