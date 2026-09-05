# src/agent_from_scratch/agent_response.py
import json
import logging
from openai import OpenAI
from src.agent_from_scratch.tool_manager import ToolManager
from src.common.config import DEFAULT_MODEL, SYSTEM_PROMPT, check_api_key, DEFAULT_MAX_ITERATIONS

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

class MarketMindResponseAgent:
    """
    An AI-powered financial assistant using OpenAI's Response API with simplified state management.
    """

    def __init__(self, model=DEFAULT_MODEL):
        logger.debug(f"Initializing MarketMindResponseAgent with model: {model}")
        self.tool_manager = ToolManager()
        self.model = model
        self.client = OpenAI(api_key=check_api_key())

        # Initialize empty tool schemas
        self.tool_schemas = []

        # Track the most recent response ID for state management
        self.previous_response_id = None

        logger.debug(f"MarketMindResponseAgent initialized successfully with model: {model}")

    def register_tool(self, name, description, tool_function):
        """Register a new tool with the agent."""
        logger.debug(f"Registering tool: {name} - {description}")
        self.tool_manager.register_tool(name, description, tool_function)

        # Update tool schemas immediately - convert to Response API format
        raw_schemas = self.tool_manager.get_schema_for_tools()

        # Convert Chat Completions API format to Response API format
        response_api_tools = []
        for schema in raw_schemas:
            if schema.get("type") == "function":
                function_data = schema.get("function", {})
                response_api_tools.append({
                    "type": "function",
                    "name": function_data.get("name", ""),
                    "description": function_data.get("description", ""),
                    "parameters": function_data.get("parameters", {})
                })

        self.tool_schemas = response_api_tools
        logger.debug(f"Tool schemas updated: now have {len(self.tool_schemas)} tools")

        return self

    def _handle_function_call(self, function_call):
        """
        Handle a function call from the Response API.

        Args:
            function_call: The function call object from the Response API

        Returns:
            The result of the function call as a string
        """
        try:
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)

            logger.debug(f"Executing tool: {function_name} with args: {function_args}")

            # Execute the tool
            tool_result = self.tool_manager.execute_tool(function_name, **function_args)
            logger.debug(f"Tool execution result: {tool_result}")

            return str(tool_result)

        except Exception as e:
            error_msg = f"Error executing function call: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def process_query(self, query):
        """
        Process a user query using the Response API.

        Args:
            query: The user's query

        Returns:
            The agent's response as a string
        """
        logger.debug(f"Processing query: {query}")
        try:
            # System prompt that defines the agent's capabilities
            system_prompt = SYSTEM_PROMPT

            # Prepare the initial input for the API call
            input_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            logger.debug(f"Prepared initial input messages: {json.dumps(input_messages, indent=2)}")
            logger.debug(f"Available tools: {json.dumps(self.tool_schemas, indent=2)}")

            # Create the API call parameters
            api_params = {
                "model": self.model,
                "input": input_messages,
                "tools": self.tool_schemas
            }

            if self.previous_response_id:
                api_params["previous_response_id"] = self.previous_response_id
                logger.debug(f"Including previous response ID: {self.previous_response_id}")

            # Call the API to get the initial response
            logger.debug(f"Sending initial API request with parameters: {json.dumps(api_params, indent=2)}")
            response = self.client.responses.create(**api_params)
            logger.info(f"Received initial response with ID: {response.id}")
            logger.debug(f"Initial response output: {json.dumps(response.model_dump(), indent=2)}")

            # Save the response ID for future calls
            self.previous_response_id = response.id

            # Process the response and any subsequent responses with function calls
            iteration = 0

            # Continue processing responses until we get one without function calls
            while iteration < DEFAULT_MAX_ITERATIONS:
                iteration += 1
                logger.debug(f"Starting iteration {iteration} of response processing loop")

                # Process the current response
                output = response.output
                logger.debug(f"Processing {len(output)} output items in response {response.id}")

                # Check if we have function calls to process
                function_calls = [item for item in output if item.type == "function_call"]

                if not function_calls:
                    logger.info(f"No function calls to process in iteration {iteration}")
                    break

                logger.info(f"Found {len(function_calls)} function calls to process")

                # Process all function calls in this response
                function_call_results = []
                for idx, function_call in enumerate(function_calls):
                    logger.debug(f"Processing function call {idx + 1}/{len(function_calls)}: {function_call.name}")
                    logger.info(f"Executing function call: {function_call.name} with arguments: {function_call.arguments}")

                    # Execute the function call
                    result = self._handle_function_call(function_call)

                    # Add the result to our list
                    function_call_results.append({
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": result
                    })

                # Submit all function call results back to the API in one request
                logger.debug(f"Sending follow-up request with {len(function_call_results)} function call results")
                logger.debug(f"Function call results: {json.dumps(function_call_results, indent=2)}")

                follow_up_response = self.client.responses.create(
                    model=self.model,
                    input=function_call_results,
                    previous_response_id=response.id,
                    tools=self.tool_schemas
                )

                logger.info(f"Received follow-up response with ID: {follow_up_response.id}")
                logger.debug(f"Follow-up response output: {json.dumps(follow_up_response.model_dump(), indent=2)}")

                # Update the response ID for future calls
                self.previous_response_id = follow_up_response.id
                response = follow_up_response

            # Check if we hit the maximum number of iterations
            if iteration >= DEFAULT_MAX_ITERATIONS:
                logger.warning(f"Hit maximum iterations ({DEFAULT_MAX_ITERATIONS}) when processing function calls")

            # Return the final text response
            final_output = response.output_text
            logger.info(f"Final response text: {final_output}")
            return final_output

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"An error occurred: {str(e)}"
