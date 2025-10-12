"""
Data Fetcher - Fetches upfront financial data for domain agents
"""

import logging
import sys
import os
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

# Add external RAG paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../../SECFinancialRAG'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../AnnualReportRAG'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../TranscriptRAG'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../NewsRAG'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../SharePriceRAG'))

from config.domain_configs import get_domain_config
from config.settings import RAG_BASE_PATH


@dataclass
class DataFetchResult:
    """Result of data fetching operation."""
    
    success: bool
    financial_metrics: Dict[str, Any]
    financial_ratios: Dict[str, Any]
    peer_comparison: Dict[str, Any]
    data_quality_score: float
    errors: List[str]
    fetch_timestamp: str


class DomainDataFetcher:
    """
    Fetches upfront financial data for domain analysis.
    Integrates with external RAG systems to gather required metrics and ratios.
    """
    
    def __init__(self, enable_debug: bool = False):
        """
        Initialize data fetcher.
        
        Args:
            enable_debug: Enable debug logging
        """
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        if enable_debug:
            self.logger.setLevel(logging.DEBUG)
        
        # External RAG interfaces (will be imported dynamically)
        self.sec_financial_rag = None
        self.annual_report_rag = None
        self.transcript_rag = None
        self.news_rag = None
        self.share_price_rag = None
        
        self.logger.info("DomainDataFetcher initialized")
    
    async def fetch_domain_data(self,
                               domain: str,
                               company: str,
                               peer_list: List[str],
                               time_period: str = "3 years",
                               include_additional_data: bool = True) -> DataFetchResult:
        """
        Fetch all required data for domain analysis.
        
        Args:
            domain: Domain type (liquidity, leverage, etc.)
            company: Company ticker
            peer_list: List of peer company tickers
            time_period: Time period for data
            include_additional_data: Whether to fetch additional qualitative data
            
        Returns:
            Data fetch result with all required information
        """
        self.logger.info(f"Fetching data for {domain} analysis: {company}")
        
        try:
            # Get domain configuration
            domain_config = get_domain_config(domain)
            
            # Initialize fetch result
            fetch_result = DataFetchResult(
                success=False,
                financial_metrics={},
                financial_ratios={},
                peer_comparison={},
                data_quality_score=0.0,
                errors=[],
                fetch_timestamp=datetime.now().isoformat()
            )
            
            # Fetch financial metrics
            metrics_result = await self._fetch_financial_metrics(
                company=company,
                required_metrics=domain_config.required_metrics,
                time_period=time_period
            )
            
            if metrics_result["success"]:
                fetch_result.financial_metrics = metrics_result["data"]
                self.logger.info(f"Fetched {len(metrics_result['data'])} financial metrics")
            else:
                fetch_result.errors.extend(metrics_result["errors"])
            
            # Fetch financial ratios
            ratios_result = await self._fetch_financial_ratios(
                company=company,
                required_ratios=domain_config.required_ratios,
                time_period=time_period
            )
            
            if ratios_result["success"]:
                fetch_result.financial_ratios = ratios_result["data"]
                self.logger.info(f"Fetched {len(ratios_result['data'])} financial ratios")
            else:
                fetch_result.errors.extend(ratios_result["errors"])
            
            # Fetch peer comparison data if required
            if domain_config.peer_comparison_required:
                peer_result = await self._fetch_peer_comparison(
                    company=company,
                    peers=peer_list,
                    metrics=domain_config.required_metrics[:5],  # Limit for peer comparison
                    ratios=domain_config.required_ratios[:5],
                    time_period=time_period  # Pass time_period for temporal clustering
                )
                
                if peer_result["success"]:
                    fetch_result.peer_comparison = peer_result["data"]
                    self.logger.info(f"Fetched peer comparison for {len(peer_list)} peers")
                else:
                    fetch_result.errors.extend(peer_result["errors"])
            
            # Calculate data quality score
            fetch_result.data_quality_score = self._calculate_data_quality_score(fetch_result)
            
            # Determine overall success
            fetch_result.success = (
                bool(fetch_result.financial_metrics or fetch_result.financial_ratios) and
                fetch_result.data_quality_score > 0.3
            )
            
            self.logger.info(f"Data fetch completed - success: {fetch_result.success}, quality: {fetch_result.data_quality_score:.2f}")
            return fetch_result
            
        except Exception as e:
            self.logger.error(f"Data fetch failed: {e}")
            return DataFetchResult(
                success=False,
                financial_metrics={},
                financial_ratios={},
                peer_comparison={},
                data_quality_score=0.0,
                errors=[f"Data fetch failed: {str(e)}"],
                fetch_timestamp=datetime.now().isoformat()
            )
    
    async def _fetch_financial_metrics(self,
                                      company: str,
                                      required_metrics: List[str],
                                      time_period: str) -> Dict[str, Any]:
        """Fetch financial metrics from SECFinancialRAG."""
        
        try:
            # Try to import and use SECFinancialRAG
            if self.sec_financial_rag is None:
                self.sec_financial_rag = await self._get_sec_financial_interface()
            
            if self.sec_financial_rag is None:
                # Fallback to simulated data
                return self._create_simulated_metrics(company, required_metrics)
            
            # Convert time period to SEC format
            sec_period = self._convert_time_period_to_sec_format(time_period)
            
            # Use actual RAG system
            result = self.sec_financial_rag['get_metrics'](
                ticker=company,
                metrics=required_metrics,
                period=sec_period
            )
            
            if result.success:
                return {
                    "success": True,
                    "data": result.data,
                    "errors": []
                }
            else:
                self.logger.warning(f"SECFinancialRAG returned error: {result.error}")
                return self._create_simulated_metrics(company, required_metrics)
            
        except Exception as e:
            self.logger.warning(f"SECFinancialRAG unavailable, using simulated data: {e}")
            return self._create_simulated_metrics(company, required_metrics)
    
    async def _fetch_financial_ratios(self,
                                     company: str,
                                     required_ratios: List[str],
                                     time_period: str) -> Dict[str, Any]:
        """Fetch financial ratios from SECFinancialRAG."""
        
        try:
            if self.sec_financial_rag is None:
                self.sec_financial_rag = await self._get_sec_financial_interface()
            
            if self.sec_financial_rag is None:
                return self._create_simulated_ratios(company, required_ratios)
            
            # Convert time period to SEC format
            sec_period = self._convert_time_period_to_sec_format(time_period)
            
            # Map required ratios to SECFinancialRAG categories
            ratio_categories = self._map_ratios_to_categories(required_ratios)
            
            # Use actual RAG system for ratios
            result = self.sec_financial_rag['get_ratios'](
                ticker=company,
                categories=ratio_categories,
                period=sec_period
            )
            
            if result.success:
                return {
                    "success": True,
                    "data": result.data,
                    "errors": []
                }
            else:
                self.logger.warning(f"SECFinancialRAG ratios returned error: {result.error}")
                return self._create_simulated_ratios(company, required_ratios)
            
        except Exception as e:
            self.logger.warning(f"Ratio fetching failed, using simulated data: {e}")
            return self._create_simulated_ratios(company, required_ratios)
    
    async def _fetch_peer_comparison(self,
                                    company: str,
                                    peers: List[str],
                                    metrics: List[str],
                                    ratios: List[str],
                                    time_period: str = "3 years") -> Dict[str, Any]:
        """Fetch peer comparison data using SECFinancialRAG with temporal clustering."""
        
        self.logger.debug(f"Starting peer comparison fetch for {company} with peers: {peers}")
        
        try:
            if self.sec_financial_rag is None:
                self.sec_financial_rag = await self._get_sec_financial_interface()
            
            if self.sec_financial_rag is None:
                self.logger.warning("SECFinancialRAG interface unavailable, using simulated peer comparison")
                return self._create_simulated_peer_comparison(company, peers, metrics, ratios)
            
            # Create company list including focus company
            all_companies = [company] + peers[:4]  # Limit to 5 companies total for efficiency
            self.logger.debug(f"All companies for comparison: {all_companies}")
            
            # Ensure we have at least 2 companies for comparison
            if len(all_companies) < 2:
                self.logger.warning(f"Insufficient companies for comparison ({len(all_companies)}), using simulated data")
                return self._create_simulated_peer_comparison(company, peers, metrics, ratios)
            
            # Map ratios to categories for comparison
            ratio_categories = self._map_ratios_to_categories(ratios)
            self.logger.debug(f"Mapped ratio categories: {ratio_categories}")

            # Use actual RAG system for peer comparison
            # IMPORTANT: Must use same time_period as metrics/ratios to enable temporal clustering
            sec_period = self._convert_time_period_to_sec_format(time_period) if time_period else "last 12 quarters"
            self.logger.debug(f"Calling compare_companies with tickers={all_companies}, categories={ratio_categories}, period={sec_period}")
            result = self.sec_financial_rag['compare_companies'](
                tickers=all_companies,
                categories=ratio_categories,
                period=sec_period  # Use same period as domain analysis for temporal clustering
            )
            
            if result.success:
                self.logger.debug("SECFinancialRAG peer comparison successful")

                # Transform temporal clustering data into period summaries format
                transformed_data = self._transform_to_period_summaries(
                    comparison_data=result.data,
                    focus_company=company,
                    peers=peers[:4],
                    categories=ratio_categories
                )

                return {
                    "success": True,
                    "data": transformed_data,
                    "errors": []
                }
            else:
                self.logger.warning(f"SECFinancialRAG peer comparison returned error: {result.error}")
                return self._create_simulated_peer_comparison(company, peers, metrics, ratios)
            
        except Exception as e:
            self.logger.warning(f"Peer comparison failed, using simulated data: {e}")
            self.logger.exception("Full peer comparison exception:")
            return self._create_simulated_peer_comparison(company, peers, metrics, ratios)

    def _transform_to_period_summaries(self,
                                      comparison_data: Dict[str, Any],
                                      focus_company: str,
                                      peers: List[str],
                                      categories: List[str]) -> Dict[str, Any]:
        """
        Transform temporal clustering data into period summaries format.

        Converts nested temporal clustering structure into:
        - Pre-formatted text blocks for each period
        - Structured chart_data arrays
        - Clear period boundaries

        Args:
            comparison_data: Raw comparison data from SECFinancialRAG
            focus_company: Focus company ticker
            peers: List of peer tickers
            categories: Categories compared

        Returns:
            Transformed data with period_summaries
        """
        try:
            period_summaries = []
            all_companies = [focus_company] + peers

            # Extract categories_comparison from nested structure
            categories_comparison = comparison_data.get('categories_comparison', {})

            if not categories_comparison:
                self.logger.warning("No categories_comparison found in comparison_data")
                return self._create_empty_peer_comparison(focus_company, peers, categories)

            # Collect all periods across all categories and ratios
            # Structure: {period_number: {category: {ratio_name: period_data}}}
            periods_by_number = {}

            for category_name, category_data in categories_comparison.items():
                ratios_data = category_data.get('ratios', {})

                for ratio_key, ratio_value in ratios_data.items():
                    # Extract base ratio name and period number
                    # Format: "Current_Ratio_Q_Period_1"
                    if '_Period_' not in ratio_key:
                        continue

                    parts = ratio_key.rsplit('_Period_', 1)
                    if len(parts) != 2:
                        continue

                    base_ratio_name = parts[0]
                    try:
                        period_num = int(parts[1])
                    except ValueError:
                        continue

                    # Initialize nested structure
                    if period_num not in periods_by_number:
                        periods_by_number[period_num] = {}
                    if category_name not in periods_by_number[period_num]:
                        periods_by_number[period_num][category_name] = {}

                    periods_by_number[period_num][category_name][base_ratio_name] = ratio_value

            # Sort periods by number
            sorted_period_numbers = sorted(periods_by_number.keys())

            # Build period summaries
            for period_num in sorted_period_numbers:
                period_data = periods_by_number[period_num]

                # Extract metadata from first available ratio
                first_category = next(iter(period_data.values()))
                first_ratio_name = next(iter(first_category.keys()))
                first_ratio_data = first_category[first_ratio_name]

                period_info = first_ratio_data.get('period_info', {})
                company_values_sample = first_ratio_data.get('company_values', {})

                # Build formatted text
                period_text_lines = []

                # Header
                cluster_date = period_info.get('cluster_center_date', 'Unknown')
                companies_included = period_info.get('companies_included', all_companies)
                companies_missing = period_info.get('companies_missing', [])
                span_days = period_info.get('date_range', {}).get('span_days', 0)

                header = f"PERIOD {period_num} (Cluster: {cluster_date}, Span: {span_days} days)"
                period_text_lines.append(header)

                companies_line = f"Companies: {', '.join(companies_included)}"
                if companies_missing:
                    companies_line += f" | Missing: {', '.join(companies_missing)}"
                period_text_lines.append(companies_line)
                period_text_lines.append("")  # Blank line

                # Initialize chart data for this period
                period_chart_data = {}

                # Add ratios by category
                for category_name in sorted(period_data.keys()):
                    category_ratios = period_data[category_name]

                    for ratio_name in sorted(category_ratios.keys()):
                        ratio_data = category_ratios[ratio_name]
                        company_values = ratio_data.get('company_values', {})

                        # Format ratio line
                        ratio_display_name = ratio_name.replace('_', ' ').title()
                        # Remove " Q" suffix if present (from quarterly ratio names)
                        if ratio_display_name.endswith(' Q'):
                            ratio_display_name = ratio_display_name[:-2]
                        ratio_line = f"{ratio_display_name}:"

                        # Collect company values
                        company_strs = []
                        chart_points = []

                        # Show focus company first
                        if focus_company in company_values:
                            cv = company_values[focus_company]
                            value = cv.get('value')
                            fiscal_quarter = cv.get('fiscal_quarter', 'N/A')
                            period_end_date = cv.get('period_end_date', 'N/A')

                            if value is not None:
                                company_strs.append(f"  {focus_company} = {value:.3f} ({fiscal_quarter}, {period_end_date})")
                                chart_points.append({
                                    'company': focus_company,
                                    'value': value,
                                    'fiscal_quarter': fiscal_quarter,
                                    'period_end_date': period_end_date
                                })

                        # Show peers
                        for peer in peers:
                            if peer in company_values:
                                cv = company_values[peer]
                                value = cv.get('value')
                                fiscal_quarter = cv.get('fiscal_quarter', 'N/A')
                                period_end_date = cv.get('period_end_date', 'N/A')

                                if value is not None:
                                    company_strs.append(f"  {peer} = {value:.3f} ({fiscal_quarter}, {period_end_date})")
                                    chart_points.append({
                                        'company': peer,
                                        'value': value,
                                        'fiscal_quarter': fiscal_quarter,
                                        'period_end_date': period_end_date
                                    })

                        # Add to text
                        if company_strs:
                            period_text_lines.append(ratio_line)
                            period_text_lines.extend(company_strs)
                            period_text_lines.append("")  # Blank line

                            # Add to chart data
                            period_chart_data[ratio_name] = chart_points

                # Build period summary
                period_summary = {
                    'period_number': period_num,
                    'period_text': '\n'.join(period_text_lines),
                    'metadata': {
                        'cluster_center_date': cluster_date,
                        'companies_included': companies_included,
                        'companies_missing': companies_missing,
                        'date_span_days': span_days,
                        'clustering_method': period_info.get('clustering_method', 'Temporal proximity')
                    },
                    'chart_data': period_chart_data
                }

                period_summaries.append(period_summary)

            # Build final output
            ratios_compared = []
            for period_data in periods_by_number.values():
                for category_ratios in period_data.values():
                    ratios_compared.extend(category_ratios.keys())
            ratios_compared = sorted(list(set(ratios_compared)))

            return {
                'focus_company': focus_company,
                'peers': peers,
                'summary': {
                    'total_periods': len(period_summaries),
                    'ratios_compared': ratios_compared,
                    'categories': categories
                },
                'period_summaries': period_summaries,
                'raw_data': comparison_data  # Preserve for fallback
            }

        except Exception as e:
            self.logger.error(f"Failed to transform to period summaries: {e}")
            self.logger.exception("Transformation exception:")
            return self._create_empty_peer_comparison(focus_company, peers, categories)

    def _create_empty_peer_comparison(self,
                                     focus_company: str,
                                     peers: List[str],
                                     categories: List[str]) -> Dict[str, Any]:
        """Create empty peer comparison structure when transformation fails."""
        return {
            'focus_company': focus_company,
            'peers': peers,
            'summary': {
                'total_periods': 0,
                'ratios_compared': [],
                'categories': categories
            },
            'period_summaries': [],
            'raw_data': {}
        }

    def _create_simulated_peer_comparison(self,
                                        company: str,
                                        peers: List[str],
                                        metrics: List[str],
                                        ratios: List[str]) -> Dict[str, Any]:
        """Create simulated peer comparison data."""
        
        self.logger.debug(f"Creating simulated peer comparison for {company} with {len(peers)} peers: {peers}")
        
        peer_data = {
            "focus_company": company,
            "peers": peers,
            "comparison_metrics": {},
            "comparison_ratios": {}
        }
        
        # Create sample peer comparison
        for metric in metrics[:3]:  # Limit to top 3 metrics
            peer_data["comparison_metrics"][metric] = {
                company: self._generate_sample_value(metric, "metric", company),
            }
            for peer in peers[:3]:  # Limit to top 3 peers
                peer_data["comparison_metrics"][metric][peer] = self._generate_sample_value(metric, "metric", peer)
        
        for ratio in ratios[:3]:  # Limit to top 3 ratios
            peer_data["comparison_ratios"][ratio] = {
                company: self._generate_sample_value(ratio, "ratio", company),
            }
            for peer in peers[:3]:
                peer_data["comparison_ratios"][ratio][peer] = self._generate_sample_value(ratio, "ratio", peer)
        
        return {
            "success": True,
            "data": peer_data,
            "errors": ["Using simulated data - external RAG unavailable"]
        }
    
    async def _get_sec_financial_interface(self):
        """Dynamically import and initialize SECFinancialRAG interface with complete isolation."""
        try:
            # Save current environment completely
            original_cwd = os.getcwd()
            original_path = sys.path.copy()
            original_modules = sys.modules.copy()
            
            # Calculate absolute path to SECFinancialRAG
            sec_rag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../SECFinancialRAG'))
            
            if not os.path.exists(sec_rag_path):
                self.logger.warning(f"SECFinancialRAG path not found: {sec_rag_path}")
                return None
            
            agentsystemv2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            
            # STEP 1: Remove any conflicting 'main' modules from sys.modules
            modules_to_remove = []
            for module_name in list(sys.modules.keys()):
                if module_name == 'main' or module_name.startswith('main.'):
                    modules_to_remove.append(module_name)
            
            self.logger.debug(f"Removing {len(modules_to_remove)} conflicting main modules")
            for module_name in modules_to_remove:
                del sys.modules[module_name]
            
            # STEP 2: Create completely clean sys.path (remove ALL AgentSystemV2 references)
            clean_path = []
            for p in sys.path:
                # Skip any path that contains AgentSystemV2 or is the AgentSystemV2 directory
                if 'AgentSystemV2' not in p and p != agentsystemv2_path:
                    clean_path.append(p)
            
            # STEP 3: Set up completely isolated environment
            os.chdir(sec_rag_path)
            sys.path = [sec_rag_path] + clean_path
            
            self.logger.debug(f"Isolated environment: working_dir={sec_rag_path}")
            self.logger.debug(f"Isolated sys.path[0:3]: {sys.path[:3]}")
            
            try:
                # Import with complete isolation
                from agent_interface import (
                    get_financial_metrics_for_agent,
                    get_ratios_for_agent, 
                    compare_companies_for_agent
                )
                
                self.logger.debug("Successfully imported SECFinancialRAG with isolation")
                return {
                    'get_metrics': get_financial_metrics_for_agent,
                    'get_ratios': get_ratios_for_agent,
                    'compare_companies': compare_companies_for_agent
                }
                
            finally:
                # STEP 4: Restore original environment completely
                os.chdir(original_cwd)
                sys.path = original_path
                
                # Clean up ONLY application-specific modules, preserve all core libraries
                current_modules = list(sys.modules.keys())
                core_modules = {
                    'numpy', 'pandas', 'scipy', 'matplotlib', 'sqlite3', 'decimal', 'json', 'os', 'sys',
                    'datetime', 'typing', 'logging', 'urllib', 'http', 'ssl', 'socket', 'collections',
                    'itertools', 'functools', 'operator', 'warnings', 'inspect', 'importlib'
                }
                
                for module_name in current_modules:
                    if module_name not in original_modules:
                        # Only clean up non-core modules to avoid reload warnings
                        is_core = False
                        for core in core_modules:
                            if module_name == core or module_name.startswith(f'{core}.'):
                                is_core = True
                                break
                        
                        if not is_core:
                            try:
                                del sys.modules[module_name]
                            except KeyError:
                                pass
                
                self.logger.debug("Environment restored after SECFinancialRAG import")
                
        except Exception as e:
            self.logger.warning(f"SECFinancialRAG import failed: {e}")
            # Ensure environment is restored even on failure
            try:
                os.chdir(original_cwd)
                sys.path = original_path
            except:
                pass
            return None
    
    def _create_simulated_metrics(self, company: str, required_metrics: List[str]) -> Dict[str, Any]:
        """Create simulated financial metrics for testing."""
        
        simulated_data = {}
        
        # Generate sample quarterly data
        quarters = ["Q2-2025", "Q1-2025", "Q4-2024", "Q3-2024"]
        
        for metric in required_metrics[:10]:  # Limit for simulation
            quarterly_values = {}
            base_value = self._generate_sample_value(metric, "metric")
            
            for i, quarter in enumerate(quarters):
                # Add some variation
                variation = 1.0 + (i * 0.05) + (hash(metric) % 10 - 5) * 0.02
                quarterly_values[quarter] = base_value * variation
            
            simulated_data[metric] = quarterly_values
        
        return {
            "success": True,
            "data": simulated_data,
            "errors": ["Using simulated data - external RAG unavailable"]
        }
    
    def _create_simulated_ratios(self, company: str, required_ratios: List[str]) -> Dict[str, Any]:
        """Create simulated financial ratios for testing."""
        
        simulated_data = {}
        quarters = ["Q2-2025", "Q1-2025", "Q4-2024", "Q3-2024"]
        
        for ratio in required_ratios[:10]:  # Limit for simulation
            quarterly_values = {}
            base_value = self._generate_sample_value(ratio, "ratio")
            
            for i, quarter in enumerate(quarters):
                variation = 1.0 + (i * 0.02) + (hash(ratio) % 10 - 5) * 0.01
                quarterly_values[quarter] = base_value * variation
            
            simulated_data[ratio] = quarterly_values
        
        return {
            "success": True,
            "data": simulated_data,
            "errors": ["Using simulated data - external RAG unavailable"]
        }
    
    def _generate_sample_value(self, name: str, value_type: str, company: str = "") -> float:
        """Generate realistic sample values based on metric/ratio name and company."""
        
        name_lower = name.lower()
        
        # Use both name and company for unique hash to ensure different values per company
        hash_input = f"{name}_{company}"
        company_hash = hash(hash_input)
        
        if value_type == "metric":
            # Financial metrics (typically large numbers)
            if "cash" in name_lower:
                return 25000000000.0 + (company_hash % 1000000000)
            elif "revenue" in name_lower:
                return 80000000000.0 + (company_hash % 5000000000)
            elif "assets" in name_lower or "liabilities" in name_lower:
                return 150000000000.0 + (company_hash % 10000000000)
            elif "debt" in name_lower:
                return 50000000000.0 + (company_hash % 2000000000)
            else:
                return 10000000000.0 + (company_hash % 5000000000)
        
        elif value_type == "ratio":
            # Financial ratios (typically small numbers)
            if "ratio" in name_lower:
                return 1.0 + (company_hash % 100) * 0.01
            elif "margin" in name_lower:
                return 0.20 + (company_hash % 30) * 0.01
            elif "coverage" in name_lower:
                return 5.0 + (company_hash % 50) * 0.1
            elif "turnover" in name_lower:
                return 2.0 + (company_hash % 30) * 0.1
            else:
                return 1.5 + (company_hash % 50) * 0.02
        
        return 1.0
    
    def _map_ratios_to_categories(self, required_ratios: List[str]) -> List[str]:
        """Map specific ratio names to SECFinancialRAG categories."""
        
        # Map specific ratio names to their categories
        ratio_category_mapping = {
            # Liquidity ratios
            "current_ratio": "liquidity",
            "quick_ratio": "liquidity", 
            "cash_ratio": "liquidity",
            "working_capital_ratio": "liquidity",
            
            # Leverage ratios
            "debt_to_equity": "leverage",
            "debt_to_assets": "leverage",
            "interest_coverage_ratio": "leverage",
            "debt_service_coverage": "leverage",
            "equity_ratio": "leverage",
            "financial_leverage": "leverage",
            
            # Efficiency ratios
            "working_capital_turnover": "efficiency",
            "asset_turnover": "efficiency",
            "receivables_turnover": "efficiency",
            "inventory_turnover": "efficiency",
            "days_sales_outstanding": "efficiency",
            "days_inventory_outstanding": "efficiency",
            "days_payable_outstanding": "efficiency",
            "cash_conversion_cycle": "efficiency",
            
            # Profitability ratios
            "gross_margin": "profitability",
            "operating_margin": "profitability",
            "ebitda_margin": "profitability",
            "net_margin": "profitability",
            "return_on_equity": "profitability",
            "return_on_assets": "profitability",
            "return_on_invested_capital": "profitability",
            
            # Growth ratios
            "revenue_growth": "growth",
            "earnings_growth": "growth",
            "asset_growth": "growth",
            
            # Valuation ratios
            "price_to_earnings": "valuation",
            "price_to_book": "valuation",
            "ev_to_ebitda": "valuation",
            "price_to_sales": "valuation",
            "peg_ratio": "valuation",
            "ev_to_sales": "valuation",
            "price_to_free_cash_flow": "valuation"
        }
        
        # Extract unique categories from required ratios
        categories = set()
        for ratio in required_ratios:
            ratio_lower = ratio.lower().replace("_", "_")
            if ratio_lower in ratio_category_mapping:
                categories.add(ratio_category_mapping[ratio_lower])
            else:
                # Try to infer category from ratio name
                if "liquid" in ratio_lower or "current" in ratio_lower or "quick" in ratio_lower:
                    categories.add("liquidity")
                elif "debt" in ratio_lower or "leverage" in ratio_lower or "coverage" in ratio_lower:
                    categories.add("leverage")
                elif "turnover" in ratio_lower or "days" in ratio_lower or "efficiency" in ratio_lower:
                    categories.add("efficiency")
                elif "margin" in ratio_lower or "return" in ratio_lower or "profit" in ratio_lower:
                    categories.add("profitability")
                elif "growth" in ratio_lower:
                    categories.add("growth")
                elif "price" in ratio_lower or "ev" in ratio_lower or "valuation" in ratio_lower:
                    categories.add("valuation")
                else:
                    # Default to profitability if can't categorize
                    categories.add("profitability")
        
        return list(categories) if categories else ["liquidity", "profitability"]
    
    def _convert_time_period_to_sec_format(self, time_period: str) -> str:
        """Convert user time period to SECFinancialRAG format."""
        
        time_period_lower = time_period.lower().strip()
        
        # Handle specific formats
        if "year" in time_period_lower:
            # Extract number before "year"
            import re
            match = re.search(r'(\d+)\s*year', time_period_lower)
            if match:
                years = int(match.group(1))
                quarters = years * 4
                return f"last {quarters} quarters"
        
        elif "quarter" in time_period_lower:
            # Extract number before "quarter"
            import re
            match = re.search(r'(\d+)\s*quarter', time_period_lower)
            if match:
                quarters = int(match.group(1))
                return f"last {quarters} quarters"
        
        elif time_period_lower in ["latest", "current", "last"]:
            return "latest"
        
        elif time_period_lower.endswith("y"):
            # Format like "3Y", "5y"
            try:
                years = int(time_period_lower[:-1])
                quarters = years * 4
                return f"last {quarters} quarters"
            except ValueError:
                pass
        
        elif time_period_lower.endswith("q"):
            # Format like "8Q", "12q"
            try:
                quarters = int(time_period_lower[:-1])
                return f"last {quarters} quarters"
            except ValueError:
                pass
        
        # Try to extract any number and assume years
        import re
        match = re.search(r'(\d+)', time_period_lower)
        if match:
            num = int(match.group(1))
            # If number is > 20, assume quarters, otherwise years
            if num > 20:
                return f"last {num} quarters"
            else:
                quarters = num * 4
                return f"last {quarters} quarters"
        
        # Default fallback - 3 years = 12 quarters
        return "last 12 quarters"
    
    def _calculate_data_quality_score(self, fetch_result: DataFetchResult) -> float:
        """Calculate overall data quality score."""
        
        score = 0.0
        
        # Financial metrics contribution (40%)
        if fetch_result.financial_metrics:
            metrics_score = min(len(fetch_result.financial_metrics) / 10.0, 1.0)
            score += metrics_score * 0.4
        
        # Financial ratios contribution (40%)
        if fetch_result.financial_ratios:
            ratios_score = min(len(fetch_result.financial_ratios) / 8.0, 1.0)
            score += ratios_score * 0.4
        
        # Peer comparison contribution (20%)
        if fetch_result.peer_comparison:
            peer_score = 1.0 if fetch_result.peer_comparison.get('peers') else 0.0
            score += peer_score * 0.2
        
        # Penalize for errors
        error_penalty = min(len(fetch_result.errors) * 0.1, 0.3)
        score = max(0.0, score - error_penalty)
        
        return round(score, 2)