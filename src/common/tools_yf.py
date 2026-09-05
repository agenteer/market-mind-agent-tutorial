# src/common/tools_yf.py
import logging
import sys
import yfinance as yf
from datetime import datetime, timedelta

# Set up a colorful console handler for nice log presentation
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            # Colorize function calls
            if record.msg.startswith("Getting stock price"):
                return f"  🔍 {record.getMessage()}"
            elif record.msg.startswith("Getting stock history"):
                return f"  📈 {record.getMessage()}"
            elif record.msg.startswith("Getting company info"):
                return f"  🏢 {record.getMessage()}"
            elif record.msg.startswith("Getting financial metrics"):
                return f"  💰 {record.getMessage()}"
        return super().format(record)

# Configure logger for this module to output INFO level with color
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a special handler for colorful console output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter('%(message)s'))
logger.addHandler(console_handler)


def get_stock_price(ticker: str) -> str:
    """
    Get the current price of a stock.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')

    Returns:
        A string with the current price of the stock
    """
    logger.info(f"Getting stock price for {ticker}")
    ticker = ticker.upper()

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        hist = stock.history(period="1d")

        if hist.empty:
            return f"Could not retrieve stock price for {ticker}. Please check the ticker symbol."

        current_price = hist['Close'].iloc[-1]

        try:
            prev_close = info.get('previousClose', hist['Open'].iloc[-1])
            change = current_price - prev_close
            percent_change = (change / prev_close) * 100
        except (KeyError, IndexError):
            change = 0
            percent_change = 0

        # Get company name
        company_name = info.get('shortName', ticker)

        # Format with direction indicators
        direction = "up" if change > 0 else "down" if change < 0 else "unchanged"

        return f"{company_name} ({ticker}) is currently trading at ${current_price:.2f}, {direction} {abs(percent_change):.2f}% today."
    except Exception as e:
        logger.error(f"Error fetching stock price for {ticker}: {str(e)}", exc_info=True)
        return f"An error occurred while fetching the stock price for {ticker}: {str(e)}"


def get_stock_history(ticker: str, days: int) -> str:
    """
    Get historical price data for a stock using Yahoo Finance.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')
        days: Number of days of history to retrieve (default: 7)

    Returns:
        A string with the historical price information
    """
    logger.info(f"Getting {days} days of stock history for {ticker}")
    ticker = ticker.upper()

    try:
        # Get ticker info from Yahoo Finance
        stock = yf.Ticker(ticker)

        # Get historical data for the specified period
        # Add a buffer to ensure we get enough business days
        buffer_days = max(5, int(days * 1.5))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=buffer_days)

        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return f"Could not retrieve historical data for {ticker}."

        # Filter to the exact number of days requested
        hist = hist.tail(days)

        # Format the historical data
        history_lines = []
        for date, row in hist.iterrows():
            date_str = date.strftime("%Y-%m-%d")
            close_price = row['Close']
            history_lines.append(f"{date_str}: ${close_price:.2f}")

        # Get company name
        company_name = stock.info.get('shortName', ticker)

        result = f"Historical prices for {company_name} ({ticker}) - last {len(history_lines)} days:\n\n"
        result += "\n".join(history_lines)

        return result

    except Exception as e:
        logger.error(f"Error fetching stock history for {ticker}: {str(e)}", exc_info=True)
        return f"Error retrieving historical data for {ticker}: {str(e)}"


def get_company_info(ticker: str) -> str:
    """
    Get basic information about a company using Yahoo Finance.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')

    Returns:
        A string with the company information
    """
    logger.info(f"Getting company info for {ticker}")
    ticker = ticker.upper()

    try:
        # Get ticker info from Yahoo Finance
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return f"Company information for {ticker} is not available."

        # Extract relevant company information
        company_name = info.get('shortName', ticker)
        sector = info.get('sector', 'Not available')
        industry = info.get('industry', 'Not available')
        description = info.get('longBusinessSummary', 'No description available')

        # Additional information if available
        website = info.get('website', 'Not available')
        employees = info.get('fullTimeEmployees', 'Not available')
        country = info.get('country', 'Not available')
        city = info.get('city', 'Not available')

        # Format the company profile
        result = f"Company Profile: {company_name} ({ticker})\n\n"
        result += f"Sector: {sector}\n"
        result += f"Industry: {industry}\n"
        result += f"Country: {country}\n"
        result += f"City: {city}\n"
        result += f"Website: {website}\n"
        result += f"Employees: {employees}\n\n"
        result += f"Description: {description}"

        return result

    except Exception as e:
        logger.error(f"Error fetching company info for {ticker}: {str(e)}", exc_info=True)
        return f"Error retrieving company information for {ticker}: {str(e)}"


def get_financial_metrics(ticker: str) -> str:
    """
    Get key financial metrics for a company using Yahoo Finance.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL')

    Returns:
        A string with key financial metrics
    """
    logger.info(f"Getting financial metrics for {ticker}")
    ticker = ticker.upper()

    try:
        # Get ticker info from Yahoo Finance
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info:
            return f"Financial metrics for {ticker} are not available."

        # Extract key financial metrics
        company_name = info.get('shortName', ticker)

        # Market data
        market_cap = info.get('marketCap', 'N/A')
        if isinstance(market_cap, (int, float)):
            market_cap = f"${market_cap / 1_000_000_000:.2f} billion"

        pe_ratio = info.get('trailingPE', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            pe_ratio = f"{pe_ratio:.2f}"

        dividend_yield = info.get('dividendYield', 'N/A')
        if isinstance(dividend_yield, (int, float)):
            # Yahoo Finance reports this field as a percentage already.
            dividend_yield = f"{dividend_yield:.2f}%"

        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 'N/A')

        # Financial metrics
        revenue = info.get('totalRevenue', 'N/A')
        if isinstance(revenue, (int, float)):
            revenue = f"${revenue / 1_000_000_000:.2f} billion"

        profit_margin = info.get('profitMargins', 'N/A')
        if isinstance(profit_margin, (int, float)):
            profit_margin = f"{profit_margin * 100:.2f}%"

        return_on_equity = info.get('returnOnEquity', 'N/A')
        if isinstance(return_on_equity, (int, float)):
            return_on_equity = f"{return_on_equity * 100:.2f}%"

        # Format the metrics
        result = f"Financial Metrics: {company_name} ({ticker})\n\n"
        result += f"Market Cap: {market_cap}\n"
        result += f"P/E Ratio: {pe_ratio}\n"
        result += f"Dividend Yield: {dividend_yield}\n"
        result += f"52-Week Range: ${fifty_two_week_low} - ${fifty_two_week_high}\n\n"
        result += f"Revenue: {revenue}\n"
        result += f"Profit Margin: {profit_margin}\n"
        result += f"Return on Equity: {return_on_equity}"

        return result

    except Exception as e:
        logger.error(f"Error fetching financial metrics for {ticker}: {str(e)}", exc_info=True)
        return f"Error retrieving financial metrics for {ticker}: {str(e)}"
