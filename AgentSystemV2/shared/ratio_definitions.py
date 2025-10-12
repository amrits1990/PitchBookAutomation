"""
Ratio Definitions Utility - Enriches final agent output with ratio definitions
"""

import logging
import sys
import os
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

# Add SECFinancialRAG path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../SECFinancialRAG'))


logger = logging.getLogger(__name__)


@dataclass
class RatioDefinition:
    """Ratio definition data structure."""
    name: str
    formula: str
    description: Optional[str] = None
    category: Optional[str] = None
    interpretation: Optional[str] = None


def enrich_report_with_ratio_definitions(
    report_output: Any,
    enable_debug: bool = False
) -> Any:
    """
    Enrich ResearchReportOutput with ratio definitions from SECFinancialRAG database.

    This function should be called AFTER all domain agent iterations are complete
    and the final ResearchReportOutput is available.

    Args:
        report_output: ResearchReportOutput object (from config.schemas)
        enable_debug: Enable debug logging

    Returns:
        Modified ResearchReportOutput with ratio_definitions field populated
    """
    if enable_debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Enriching report with ratio definitions for {report_output.company}")

    try:
        # Extract unique ratio names and companies from chartable_ratios
        ratio_names, companies = _extract_ratios_and_companies(report_output)

        if not ratio_names:
            logger.warning("No chartable ratios found in report output")
            return report_output

        logger.debug(f"Found {len(ratio_names)} unique ratios: {ratio_names}")
        logger.debug(f"Found {len(companies)} companies: {companies}")

        # Fetch ratio definitions from SECFinancialRAG
        ratio_definitions = _fetch_ratio_definitions(ratio_names, companies)

        if not ratio_definitions:
            logger.warning("No ratio definitions fetched from database")
            return report_output

        logger.info(f"Successfully fetched {len(ratio_definitions)} ratio definitions")

        # Add definitions to report output
        report_output.ratio_definitions = ratio_definitions

        return report_output

    except Exception as e:
        logger.error(f"Failed to enrich report with ratio definitions: {e}")
        logger.exception("Full exception:")
        # Return original report on error
        return report_output


def _extract_ratios_and_companies(report_output: Any) -> tuple[Set[str], Set[str]]:
    """
    Extract unique ratio names and companies from ResearchReportOutput.

    Args:
        report_output: ResearchReportOutput object

    Returns:
        Tuple of (ratio_names_set, companies_set)
    """
    ratio_names = set()
    companies = set()

    # Add focus company
    if hasattr(report_output, 'company') and report_output.company:
        companies.add(report_output.company)

    # Extract from chartable_ratios
    if hasattr(report_output, 'chartable_ratios') and report_output.chartable_ratios:
        for ratio_entry in report_output.chartable_ratios:
            # Extract ratio name
            if hasattr(ratio_entry, 'ratio_name') and ratio_entry.ratio_name:
                ratio_names.add(ratio_entry.ratio_name)

            # Extract companies from data_points
            if hasattr(ratio_entry, 'data_points') and ratio_entry.data_points:
                for data_point in ratio_entry.data_points:
                    if hasattr(data_point, 'company') and data_point.company:
                        companies.add(data_point.company)

    # Extract from chartable_metrics (if any)
    if hasattr(report_output, 'chartable_metrics') and report_output.chartable_metrics:
        for metric_entry in report_output.chartable_metrics:
            # Extract companies from data_points
            if hasattr(metric_entry, 'data_points') and metric_entry.data_points:
                for data_point in metric_entry.data_points:
                    if hasattr(data_point, 'company') and data_point.company:
                        companies.add(data_point.company)

    return ratio_names, companies


def _fetch_ratio_definitions(
    ratio_names: Set[str],
    companies: Set[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch ratio definitions from SECFinancialRAG database.

    Structure returned:
    {
        'Current_Ratio': {
            'formula': 'current_assets / current_liabilities',
            'description': 'Measures ability to pay short-term debts',
            'category': 'liquidity',
            'interpretation': 'Values above 1.0 indicate sufficient liquidity',
            'company_specific': {
                'AAPL': 'Company-specific description if different',
                'MSFT': 'Company-specific description if different'
            }
        }
    }

    Args:
        ratio_names: Set of ratio names to fetch
        companies: Set of company tickers

    Returns:
        Dictionary mapping ratio names to their definitions
    """
    definitions = {}

    try:
        # Save current working directory
        original_cwd = os.getcwd()
        original_path = sys.path.copy()

        # Navigate to SECFinancialRAG directory
        sec_rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SECFinancialRAG'))

        if not os.path.exists(sec_rag_path):
            logger.warning(f"SECFinancialRAG path not found: {sec_rag_path}")
            return definitions

        # Change to SECFinancialRAG directory
        os.chdir(sec_rag_path)
        sys.path.insert(0, sec_rag_path)

        try:
            # Import SECFinancialRAG's agent interface
            from agent_interface import get_ratio_definition_for_agent

            # Fetch definition for each ratio
            for ratio_name in ratio_names:
                try:
                    # Normalize ratio name (replace spaces with underscores)
                    normalized_ratio_name = ratio_name.replace(' ', '_')

                    # First, try to get global definition (no ticker)
                    response = get_ratio_definition_for_agent(normalized_ratio_name, ticker=None)

                    if response.success and response.data:
                        ratio_def = {
                            'formula': response.data.get('formula', ''),
                            'description': response.data.get('description', ''),
                            'category': response.data.get('category', ''),
                            'interpretation': response.data.get('interpretation', ''),
                            'company_specific': {}
                        }

                        # Check for company-specific definitions
                        for company in companies:
                            try:
                                company_response = get_ratio_definition_for_agent(normalized_ratio_name, ticker=company)
                                if company_response.success and company_response.data:
                                    company_desc = company_response.data.get('description')
                                    # Only add if different from global
                                    if company_desc and company_desc != ratio_def['description']:
                                        ratio_def['company_specific'][company] = company_desc
                            except Exception as e:
                                logger.debug(f"Could not fetch company-specific definition for {ratio_name} - {company}: {e}")
                                continue

                        definitions[ratio_name] = ratio_def
                        logger.info(f"✓ Fetched definition for '{ratio_name}': {ratio_def['formula']}")
                    else:
                        logger.warning(f"Could not fetch definition for {ratio_name}: {response.error if hasattr(response, 'error') else 'Unknown error'}")

                except Exception as e:
                    logger.warning(f"Error fetching definition for {ratio_name}: {e}")
                    continue

        finally:
            # Restore environment
            os.chdir(original_cwd)
            sys.path = original_path

    except Exception as e:
        logger.error(f"Error accessing SECFinancialRAG: {e}")
        logger.exception("Full exception:")

    return definitions


def format_ratio_definitions_for_chart(
    ratio_definitions: Dict[str, Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Format ratio definitions for chart display.

    Converts nested definition structure into a flat list suitable for
    displaying alongside charts.

    Args:
        ratio_definitions: Dictionary of ratio definitions

    Returns:
        List of formatted ratio definitions for display
    """
    formatted = []

    for ratio_name, definition in ratio_definitions.items():
        entry = {
            'ratio_name': ratio_name,
            'formula': definition.get('formula', ''),
            'description': definition.get('description', ''),
            'category': definition.get('category', ''),
            'interpretation': definition.get('interpretation', '')
        }

        # Add company-specific notes if any
        company_specific = definition.get('company_specific', {})
        if company_specific:
            notes = []
            for company, desc in company_specific.items():
                notes.append(f"{company}: {desc}")
            entry['company_specific_notes'] = '; '.join(notes)

        formatted.append(entry)

    return formatted
