"""
Utility to fetch latest Fiscal Year (FY) and Fiscal Quarter (FQ) ending periods and dates for a given company ticker.

Example output:
AAPL: latest FY 9/30/2024 FY2024 , FQ 6/30/2025 Q3 2025
"""

import sys
from AnnualReportRAG.fiscal_year_corrector import CompanyFactsHandler

def get_latest_fy_fq(ticker, user_agent=None):
    handler = CompanyFactsHandler(user_agent=user_agent)
    latest_10k, latest_10q = handler.get_latest_filings_for_search(ticker)
    
    result = {}
    if latest_10k:
        result['latest_fy'] = {
            'fiscal_year': latest_10k.get('fiscal_year'),
            'end_date': latest_10k.get('end_date'),
            'form_type': latest_10k.get('form_type')
        }
    if latest_10q:
        result['latest_fq'] = {
            'fiscal_year': latest_10q.get('fiscal_year'),
            'fiscal_quarter': latest_10q.get('fiscal_quarter'),
            'end_date': latest_10q.get('end_date'),
            'form_type': latest_10q.get('form_type')
        }
    return result

def print_latest_fy_fq(ticker, user_agent=None):
    info = get_latest_fy_fq(ticker, user_agent)
    output = f"{ticker}: "
    if 'latest_fy' in info:
        fy = info['latest_fy']
        output += f"latest FY {fy['end_date']} FY{fy['fiscal_year']}"
    if 'latest_fq' in info:
        fq = info['latest_fq']
        output += f" , FQ {fq['end_date']} Q{fq.get('fiscal_quarter', '')} {fq['fiscal_year']}"
    print(output)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get latest FY and FQ ending periods for a company ticker.")
    parser.add_argument("ticker", type=str, help="Company ticker symbol (e.g., AAPL)")
    parser.add_argument("--user-agent", type=str, default=None, help="SEC user agent (email address)")
    args = parser.parse_args()
    print_latest_fy_fq(args.ticker, args.user_agent)
