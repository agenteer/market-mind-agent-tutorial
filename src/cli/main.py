# src/cli/main.py
import asyncio
import click
import logging
from src.openai_agent_sdk.agent import MarketMindOpenAIAgent
from src.agent_from_scratch.agent_chat import MarketMindChatAgent
from src.agent_from_scratch.agent_response import MarketMindResponseAgent
from src.common.config import check_api_key, setup_logging, DEFAULT_MODEL, DEFAULT_DEBUG_MODULES
from src.common.tools_yf import (
    get_stock_price,
    get_stock_history,
    get_company_info,
    get_financial_metrics
)

# Get logger for this module
logger = logging.getLogger(__name__)


def require_openai_api_key():
    """Raise a concise command-line error before starting an agent."""
    try:
        check_api_key()
    except ValueError as error:
        raise click.ClickException(str(error)) from error

@click.group(context_settings=dict(help_option_names=['-h', '--help']))
def cli():
    """MarketMind: Your AI-powered financial assistant."""
    pass

@cli.command()
@click.option('--model', default=DEFAULT_MODEL, help='The model to use for the agent')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--use-explicit-memory/--no-explicit-memory', default=True,
              help='Use explicit conversation memory in the system prompt')
@click.option('--use-response-id/--no-response-id', default=True,
              help='Use OpenAI response IDs for conversation continuity')
def openai_agent_sdk(model, debug, use_explicit_memory, use_response_id):
    """Start MarketMind using OpenAI Agent SDK."""

    require_openai_api_key()

    # Set up logging - always log to file if debug is enabled, never to console for CLI
    log_filename = setup_logging(
        debug=debug,
        module_loggers=DEFAULT_DEBUG_MODULES,
        log_to_file=debug,
        console_output=False  # Don't output logs to console for CLI apps
    )

    logger.info(f"Starting MarketMind with model={model}, explicit_memory={use_explicit_memory}, response_id={use_response_id}")

    # Initialize the agent with memory options
    agent = MarketMindOpenAIAgent(
        model=model,
        use_explicit_memory=use_explicit_memory,
        use_response_id_memory=use_response_id
    )

    click.echo(click.style("\n🤖 MarketMind Financial Assistant powered by OpenAI Agent SDK", fg='blue', bold=True))
    click.echo(click.style("Ask me about stocks, companies, or financial metrics. Type 'exit' to quit.\n", fg='blue'))

    # Display active memory settings
    memory_settings = []
    if use_explicit_memory:
        memory_settings.append("conversation history in system prompt")
    if use_response_id:
        memory_settings.append("OpenAI response ID continuity")

    if memory_settings:
        click.echo(click.style(f"Memory enabled: {', '.join(memory_settings)}", fg='yellow'))
    else:
        click.echo(click.style("Memory disabled: Agent has no conversational context", fg='yellow'))

    if log_filename:
        click.echo(click.style(f"Log file: {log_filename}", fg='yellow'))

    # Use this function to create the event loop and run the conversation
    async def run_conversation():
        # Main conversation loop
        while True:
            # Get user input
            user_input = click.prompt(click.style("You", fg='green', bold=True))

            # Check for exit command
            if user_input.lower() in ('exit', 'quit', 'q'):
                logger.info("User requested exit")
                click.echo(click.style("\nThank you for using MarketMind! Goodbye.", fg='blue'))
                break

            # Process the query
            click.echo(click.style("MarketMind", fg='blue', bold=True) + " is thinking...")

            click.echo(click.style("  🤔 Processing query and deciding on actions...", fg="yellow"))

            try:
                # Process the query using the agent - it now automatically handles response IDs
                response = await agent.process_query(user_input)
                click.echo(click.style("  ✅ Analysis complete, generating response...", fg="green"))

                # Display the response
                click.echo(click.style("MarketMind", fg='blue', bold=True) + f": {response}\n")

                # Log memory stats for debugging
                if use_explicit_memory:
                    memory_size = len(agent.context.memory.conversation_history) if agent.context.memory.conversation_history else 0
                    logger.debug(f"Conversation memory size: {memory_size} interactions")

                if use_response_id and agent.context.previous_response_id:
                    logger.debug(f"Response ID captured for conversation continuity")

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                click.echo(click.style("  ❌ Error processing query", fg="red"))
                click.echo(click.style("MarketMind", fg='blue', bold=True) +
                          f": I encountered an error while processing your request. Please try again.\n")

    # Run the async conversation loop
    asyncio.run(run_conversation())


@cli.command()
@click.option('--model', default=DEFAULT_MODEL, help='The model to use for the agent')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def chat_completion(model, debug):
    """Start MarketMind using Chat Completion API."""

    require_openai_api_key()

    # Set up logging - always log to file if debug is enabled, never to console for CLI
    log_filename = setup_logging(
        debug=debug,
        module_loggers=DEFAULT_DEBUG_MODULES,
        log_to_file=debug,
        console_output=False  # Don't output logs to console for CLI apps
    )

    logger.info(f"Starting MarketMind Chat Completion Agent with model={model}")

    # Initialize the agent
    agent = MarketMindChatAgent(model=model)

    # Register all the tools
    agent.register_tool(
        "get_stock_price",
        "Get the current price of a stock",
        get_stock_price
    )

    agent.register_tool(
        "get_stock_history",
        "Get historical price data for a stock",
        get_stock_history
    )

    agent.register_tool(
        "get_company_info",
        "Get basic information about a company",
        get_company_info
    )

    agent.register_tool(
        "get_financial_metrics",
        "Get key financial metrics for a company",
        get_financial_metrics
    )

    click.echo(click.style("\n🤖 MarketMind Financial Assistant powered by Chat Completion API", fg='blue', bold=True))
    click.echo(click.style("Ask me about stocks, companies, or financial metrics. Type 'exit' to quit.\n", fg='blue'))

    if log_filename:
        click.echo(click.style(f"Log file: {log_filename}", fg='yellow'))

    # Main conversation loop
    while True:
        # Get user input
        user_input = click.prompt(click.style("You", fg='green', bold=True))

        # Check for exit command
        if user_input.lower() in ('exit', 'quit', 'q'):
            logger.info("User requested exit")
            click.echo(click.style("\nThank you for using MarketMind! Goodbye.", fg='blue'))
            break

        # Process the query
        click.echo(click.style("MarketMind", fg='blue', bold=True) + " is thinking...")

        click.echo(click.style("  🤔 Processing query and deciding on actions...", fg="yellow"))

        try:
            # Process the query
            response = agent.process_query(user_input)
            click.echo(click.style("  ✅ Analysis complete, generating response...", fg="green"))

            # Display the response
            click.echo(click.style("MarketMind", fg='blue', bold=True) + f": {response}\n")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            click.echo(click.style("  ❌ Error processing query", fg="red"))
            click.echo(click.style("MarketMind", fg='blue', bold=True) +
                      f": I encountered an error while processing your request. Please try again.\n")


@cli.command()
@click.option('--model', default=DEFAULT_MODEL, help='The model to use for the agent')
@click.option('--debug', is_flag=True, help='Enable debug logging')
def response_api(model, debug):
    """Start MarketMind using the OpenAI Response API."""

    require_openai_api_key()

    # Set up logging - always log to file if debug is enabled, never to console for CLI
    log_filename = setup_logging(
        debug=debug,
        module_loggers=DEFAULT_DEBUG_MODULES,
        log_to_file=debug,
        console_output=False  # Don't output logs to console for CLI apps
    )

    logger.info(f"Starting MarketMind Response API Agent with model={model}")

    # Initialize the agent
    agent = MarketMindResponseAgent(model=model)

    # Register all the tools
    agent.register_tool(
        "get_stock_price",
        "Get the current price of a stock",
        get_stock_price
    )

    agent.register_tool(
        "get_stock_history",
        "Get historical price data for a stock",
        get_stock_history
    )

    agent.register_tool(
        "get_company_info",
        "Get basic information about a company",
        get_company_info
    )

    agent.register_tool(
        "get_financial_metrics",
        "Get key financial metrics for a company",
        get_financial_metrics
    )

    click.echo(click.style("\n🤖 MarketMind Financial Assistant powered by Response API", fg='blue', bold=True))
    click.echo(click.style("Ask me about stocks, companies, or financial metrics. Type 'exit' to quit.\n", fg='blue'))

    if log_filename:
        click.echo(click.style(f"Log file: {log_filename}", fg='yellow'))

    # Main conversation loop
    while True:
        # Get user input
        user_input = click.prompt(click.style("You", fg='green', bold=True))

        # Check for exit command
        if user_input.lower() in ('exit', 'quit', 'q'):
            logger.info("User requested exit")
            click.echo(click.style("\nThank you for using MarketMind! Goodbye.", fg='blue'))
            break

        # Process the query
        click.echo(click.style("MarketMind", fg='blue', bold=True) + " is thinking...")

        click.echo(click.style("  🤔 Processing query and deciding on actions...", fg="yellow"))

        try:
            # Process the query
            response = agent.process_query(user_input)
            click.echo(click.style("  ✅ Analysis complete, generating response...", fg="green"))

            # Display the response
            click.echo(click.style("MarketMind", fg='blue', bold=True) + f": {response}\n")

            # Log response ID for debugging
            if agent.previous_response_id:
                logger.debug(f"Response ID captured for conversation continuity: {agent.previous_response_id}")

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            click.echo(click.style("  ❌ Error processing query", fg="red"))
            click.echo(click.style("MarketMind", fg='blue', bold=True) +
                      f": I encountered an error while processing your request. Please try again.\n")


def main():
    """Entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        click.echo(click.style("An unexpected error occurred. Please check the logs.", fg="red"))


if __name__ == "__main__":
    main()
