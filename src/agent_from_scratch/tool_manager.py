# src/agent_from_scratch/tool_manager.py
import json
import inspect
import logging
from typing import Any, Dict, List, Callable, Union, get_type_hints

# Configure logging
logger = logging.getLogger(__name__)

class ToolManager:
    """
    Manages the registration and execution of tools.
    """

    def __init__(self):
        self.tools = {}

    def _generate_parameter_schema(self, function: Callable) -> Dict[str, Any]:
        """
        Generate a JSON schema for the function parameters.

        Args:
            function: The function to generate a schema for

        Returns:
            A JSON schema for the function parameters
        """
        logger.debug(f"Generating parameter schema for function: {function.__name__}")
        signature = inspect.signature(function)
        type_hints = get_type_hints(function)

        logger.debug(f"Function signature: {signature}")
        logger.debug(f"Type hints: {type_hints}")

        properties = {}
        required = []

        for param_name, param in signature.parameters.items():
            # Get the parameter type from type hints, default to str if not specified
            param_type = type_hints.get(param_name, str)

            # Handle Optional types (Union[Type, None])
            if hasattr(param_type, "__origin__") and param_type.__origin__ is Union:
                # Check if this is Optional[Type] (Union[Type, None])
                args = param_type.__args__
                if len(args) == 2 and args[1] is type(None):  # noqa: E721
                    # This is Optional[Type], use the first type
                    param_type = args[0]
                    logger.debug(f"Detected Optional type for {param_name}, using {param_type}")

            # Handle both direct types and type annotations
            if hasattr(param_type, "__origin__"):
                # For annotations like List[int], Dict[str, int], etc.
                origin = param_type.__origin__
                if origin is list or origin is List:
                    json_type = "array"
                elif origin is dict or origin is Dict:
                    json_type = "object"
                else:
                    # Default to the name of the origin
                    json_type = origin.__name__.lower()
                    logger.debug(f"Using origin name for {param_name}: {json_type}")
            else:
                # For direct types like int, str, etc.
                param_type_name = param_type.__name__

                # Map Python types to JSON schema types
                type_map = {
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean",
                    "list": "array",
                    "dict": "object"
                }

                json_type = type_map.get(param_type_name, "string")
                logger.debug(f"Mapped {param_type_name} to {json_type} for {param_name}")

            # Extract parameter description from docstring if available
            param_desc = f"Parameter {param_name} for {function.__name__}"
            if function.__doc__:
                # Look for Args section in docstring
                doc_lines = function.__doc__.split("\n")
                in_args_section = False
                for line in doc_lines:
                    line = line.strip()
                    if line.startswith("Args:"):
                        in_args_section = True
                        continue
                    if in_args_section and line.startswith(param_name + ":"):
                        param_desc = line[len(param_name + ":"):].strip()
                        break
                    # If we hit a new section, stop looking
                    if in_args_section and line.endswith(":") and not line.startswith(param_name):
                        break

            properties[param_name] = {
                "type": json_type,
                "description": param_desc
            }

            # If the parameter has no default value, it's required
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
                logger.debug(f"Parameter {param_name} is required")

        schema = {
            "type": "object",
            "properties": properties,
            "required": required
        }

        logger.debug(f"Generated schema: {json.dumps(schema, indent=2)}")
        return schema

    def register_tool(self, name, description, tool_function):
        """
        Register a new tool with the manager.

        Args:
            name: The unique name of the tool
            description: A description of what the tool does
            tool_function: The function that implements the tool
        """
        logger.debug(f"Registering tool: {name} - {description}")

        # Generate parameter schema at registration time
        parameter_schema = self._generate_parameter_schema(tool_function)

        self.tools[name] = {
            "description": description,
            "function": tool_function,
            "schema": parameter_schema
        }

        logger.debug(f"Tool registered successfully: {name}")
        return self  # Allow method chaining

    def get_tool(self, name):
        """Get a tool by name."""
        return self.tools.get(name, {}).get("function")

    def list_tools(self):
        """List all available tools with their descriptions."""
        return {name: info["description"] for name, info in self.tools.items()}

    def get_schema_for_tools(self):
        """
        Get all tools in the schema expected by tool call API.

        Returns:
            A list of tool schema definitions
        """
        logger.debug("Preparing tool schema definitions")
        tools = []

        for name, info in self.tools.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["schema"]
                }
            })

        logger.debug(f"Prepared {len(tools)} tool schema")
        return tools

    def execute_tool(self, name, **kwargs):
        """
        Execute a tool by name with the provided arguments.

        Args:
            name: The name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            The result of the tool execution, or an error message if the tool doesn't exist
        """
        tool_function = self.get_tool(name)
        if not tool_function:
            error_msg = f"Error: Tool '{name}' not found"
            logger.error(error_msg)
            return error_msg

        try:
            logger.debug(f"Executing tool '{name}' with args: {kwargs}")
            result = tool_function(**kwargs)
            logger.debug(f"Tool '{name}' executed successfully with result: {result}")
            return result
        except Exception as e:
            error_msg = f"Error executing tool '{name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
