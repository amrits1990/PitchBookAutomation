"""
Example test for LatestFYFQEnding agent interface utility.
User provides a ticker, receives latest FY and FQ results.
"""

from LatestFYFQEnding.agent_interface import agent_entry

def test_agent_interface():
    # Example tickers to test
    tickers = ["AAPL", "MSFT", "TGT", "INVALIDTICKER"]
    user_agent = "testuser@example.com"  # Replace with your SEC user agent email
    for ticker in tickers:
        print(f"Testing ticker: {ticker}")
        result = agent_entry(ticker, user_agent=user_agent)
        print(result)
        print("-" * 60)

if __name__ == "__main__":
    test_agent_interface()
