"""
Agent interface utility to fetch latest Fiscal Year (FY) and Fiscal Quarter (FQ) ending periods and dates for a given company ticker.

Provides robust error handling and a callable interface for agent integration.

Example output:
AAPL: latest FY 9/30/2024 FY2024 , FQ 6/30/2025 Q3 2025
"""

from AnnualReportRAG.fiscal_year_corrector import CompanyFactsHandler
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class LatestFYFQAgentInterface:
    """
    Agent interface for retrieving latest FY and FQ ending periods and dates for a company ticker.
    """
    def __init__(self, user_agent: Optional[str] = None):
        self.handler = CompanyFactsHandler(user_agent=user_agent)

    def get_latest_fy_fq(self, ticker: str) -> Dict[str, Any]:
        result = {'ticker': ticker}
        try:
            if not ticker or not isinstance(ticker, str):
                raise ValueError("Ticker must be a non-empty string.")
            latest_10k, latest_10q = self.handler.get_latest_filings_for_search(ticker)
            if not latest_10k and not latest_10q:
                result['error'] = f"No filings found for ticker: {ticker}. Check if ticker is valid and SEC data is available."
                return result
            if latest_10k:
                result['latest_fy'] = {
                    'fiscal_year': latest_10k.get('fiscal_year'),
                    'end_date': latest_10k.get('end_date'),
                    'form_type': latest_10k.get('form_type')
                }
            else:
                result['latest_fy'] = None
            if latest_10q:
                result['latest_fq'] = {
                    'fiscal_year': latest_10q.get('fiscal_year'),
                    'fiscal_quarter': latest_10q.get('fiscal_quarter'),
                    'end_date': latest_10q.get('end_date'),
                    'form_type': latest_10q.get('form_type')
                }
            else:
                result['latest_fq'] = None
            return result
        except Exception as e:
            logger.error(f"Error in get_latest_fy_fq for ticker {ticker}: {e}")
            result['error'] = str(e)
            return result

    def format_output(self, result: Dict[str, Any]) -> str:
        ticker = result.get('ticker', '')
        if 'error' in result:
            return f"{ticker}: ERROR - {result['error']}"
        output = f"{ticker}: "
        fy = result.get('latest_fy')
        fq = result.get('latest_fq')
        # FY output
        if fy:
            output += f"latest FY {fy.get('end_date', 'N/A')} FY{fy.get('fiscal_year', 'N/A')}"
        else:
            output += "No FY data found"
        # FQ output
        if fq:
            fq_end = fq.get('end_date', 'N/A')
            fq_quarter = fq.get('fiscal_quarter', 'N/A')
            fq_year = fq.get('fiscal_year', 'N/A')
            # Ensure fq_quarter is just a number (e.g., '3' from 'Q3')
            if isinstance(fq_quarter, str) and fq_quarter.startswith('Q'):
                fq_quarter_num = fq_quarter[1:]
            else:
                fq_quarter_num = fq_quarter
            # If FQ end date matches FY end date, use FY year and correct quarter
            if fy and fq_end == fy.get('end_date'):
                if fq_quarter_num == 'N/A':
                    fq_quarter_num = '4'
                output += f" , FQ {fq_end} FQ{fq_quarter_num} {fy.get('fiscal_year', fq_year)}"
            else:
                output += f" , FQ {fq_end} FQ{fq_quarter_num} {fq_year}"
        else:
            output += " , No FQ data found"
        return output

# CLI and agent entry point
def agent_entry(ticker: str, user_agent: Optional[str] = None) -> str:
    """
    Agent callable entry point. Returns formatted string for agent consumption.
    """
    interface = LatestFYFQAgentInterface(user_agent=user_agent)
    result = interface.get_latest_fy_fq(ticker)
    return interface.format_output(result)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description="Agent tool: Get latest FY and FQ ending periods for a company ticker.")
    parser.add_argument("ticker", type=str, help="Company ticker symbol (e.g., AAPL)")
    parser.add_argument("--user-agent", type=str, default=None, help="SEC user agent (email address)")
    args = parser.parse_args()
    print(agent_entry(args.ticker, args.user_agent))
