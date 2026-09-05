"""
Centralized configuration module for MarketMind Agent.

This module provides centralized configuration for the entire application including:
1. Model defaults
2. System prompts
3. API key management
4. Logging configuration
"""

# Model configuration
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_MAX_ITERATIONS = 10

# System prompts
SYSTEM_PROMPT = """
You are MarketMind, a helpful financial assistant.

You have access to the following tools:
- get_stock_price: Retrieve the latest trading price for a stock ticker.
- get_stock_history: Get historical closing prices for a stock over a specified number of days.
- get_company_info: Provide a company profile, including sector, industry, and description.
- get_financial_metrics: Display key financial metrics such as market cap, P/E ratio, and revenue.

For any question about stock prices, company information, or financial metrics, always use the appropriate tool to get the most accurate and up-to-date information.

Do not guess or fabricate financial data; rely on the tools for factual answers.

If a question is outside your tools' scope, answer to the best of your ability or politely decline.

Always respond in a clear, helpful manner.
"""

# API Key management
import os
from dotenv import load_dotenv

load_dotenv()

def check_api_key():
    """Check if the OpenAI API key is set and return it.

    Returns:
        str: The OpenAI API key

    Raises:
        ValueError: If the API key is not set
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in your environment or .env file "
            "and replace the placeholder value."
        )
    return OPENAI_API_KEY

# Logging configuration
import logging
import os
from datetime import datetime

def setup_logging(debug=False, module_loggers=None, log_to_file=False, console_output=False):
    """Set up logging configuration for the application.

    This function configures logging for the entire application. It supports:
    - Debug/Info log levels
    - File and/or console output
    - Module-specific logging levels
    - Silencing noisy third-party libraries

    Args:
        debug: Whether to enable debug logging (more verbose)
        module_loggers: List of module names to set to DEBUG level
        log_to_file: Whether to log to a file (creates a timestamped log file)
        console_output: Whether to output logs to console (should be False for CLI apps)

    Returns:
        log_filename: Path to the log file if enabled, None otherwise
    """
    # Determine log level based on debug flag
    root_level = logging.DEBUG if debug else logging.INFO

    # Configure basic logging
    handlers = []

    # Add console handler only if explicitly requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(root_level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(console_handler)

    # Add file handler if requested
    log_filename = None
    if log_to_file:
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(log_dir, f'marketmind_{timestamp}.log')

        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(root_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=root_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )

    # Silence specific third-party loggers to reduce noise in CLI applications
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Set specific module loggers
    if module_loggers:
        for module in module_loggers:
            logging.getLogger(module).setLevel(logging.DEBUG)

    return log_filename

# Default logging levels for specific modules
DEFAULT_DEBUG_MODULES = [
    'src.openai_agent_sdk',
    'src.agent_from_scratch',
    'src.cli',
    'openai.agents'
]
