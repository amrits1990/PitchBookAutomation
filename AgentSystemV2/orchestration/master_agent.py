"""
Master Agent - Coordinates multiple domain agents and provides unified interface
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from agents.domain_agent import DomainAgent
from orchestration.agent_factory import AgentFactory
from orchestration.data_fetcher import DomainDataFetcher
from tools.chat_interface import ChatInterface
from config.schemas import AnalysisInput, ResearchReportOutput, ReviewResult, MasterAnalysisRequest
from config.domain_configs import get_available_domains
from config.settings import (
    MASTER_MODEL_ID, MAX_COST_PER_ANALYSIS, DEFAULT_MAX_LOOPS,
    ENABLE_COST_TRACKING, ENABLE_BUDGET_GUARDS
)
from shared.ratio_definitions import enrich_report_with_ratio_definitions
from config.llm_cost_tracker import format_cost_summary

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.memory import MemoryManager
from config.settings import OPENROUTER_API_KEY


class MasterAgent:
    """
    Master Agent that coordinates multiple domain agents and provides unified interface.
    
    Capabilities:
    1. Run individual domain analysis
    2. Run parallel multi-domain analysis
    3. Provide unified chat interface
    4. Coordinate cost and quality optimization
    """
    
    def __init__(self, enable_debug: bool = False):
        """
        Initialize master agent.
        
        Args:
            enable_debug: Enable debug logging
        """
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        if enable_debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Initialize components
        self.agent_factory = AgentFactory(enable_debug=enable_debug)
        self.data_fetcher = DomainDataFetcher(enable_debug=enable_debug)
        self.chat_interface = ChatInterface(enable_debug=enable_debug)
        
        # Master agent for strategic coordination
        self.master_agent = self._create_master_agent()
        
        # Execution tracking
        self.execution_history: List[Dict[str, Any]] = []
        self.active_analyses: Dict[str, Dict[str, Any]] = {}
        self.cost_tracker = CostTracker() if ENABLE_COST_TRACKING else None
        
        self.logger.info("MasterAgent initialized")
    
    def _create_master_agent(self) -> Agent:
        """Create master coordination agent with agno memory."""
        
        return Agent(
            name="financial_master_agent",
            model=OpenRouter(
                id=MASTER_MODEL_ID,
                api_key=OPENROUTER_API_KEY
            ),
            memory_manager=MemoryManager(),
            instructions="""
            You are the Master Financial Analysis Agent coordinating domain-specific agents.
            
            CORE RESPONSIBILITIES:
            - Coordinate multiple domain analyses for comprehensive reports
            - Optimize analysis flow and resource allocation
            - Provide strategic oversight and quality assurance
            - Manage cost constraints and performance targets
            - Synthesize multi-domain insights into unified recommendations
            
            COORDINATION APPROACH:
            - Plan analysis sequences for maximum efficiency
            - Monitor domain agent performance and quality
            - Identify cross-domain insights and dependencies
            - Provide executive-level synthesis and recommendations
            - Ensure cost-effective resource utilization
            
            MEMORY USAGE:
            - Track analysis patterns and performance metrics
            - Remember successful domain combinations and approaches
            - Maintain context across multi-session analyses
            - Learn from user preferences and feedback patterns
            """,
            debug_mode=self.enable_debug,
            markdown=True
        )
    
    async def run_single_domain_analysis(self,
                                        company: str,
                                        domain: str,
                                        peers: List[str] = None,
                                        user_focus: str = "comprehensive analysis",
                                        time_period: str = "3 years",
                                        persona: str = "institutional_investor",
                                        max_loops: int = None) -> Dict[str, Any]:
        """
        Run analysis for a single domain.

        Args:
            company: Company ticker
            domain: Domain type (liquidity, leverage, etc.)
            peers: List of peer companies
            user_focus: User's specific focus
            time_period: Analysis time period
            persona: Target audience (banker, cfo, trader, institutional_investor, retail_investor)
            max_loops: Override default max loops

        Returns:
            Complete single domain analysis results
        """
        run_id = f"single_{domain}_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting single domain analysis: {run_id}")
        
        # Store master agent context
        await self.master_agent.arun(f"""
        Starting single domain analysis:
        Run ID: {run_id}
        Company: {company}
        Domain: {domain}
        Peers: {peers or 'None specified'}
        Focus: {user_focus}
        """)
        
        try:
            # Validate domain
            if domain not in get_available_domains():
                raise ValueError(f"Unsupported domain: {domain}")
            
            # Initialize cost tracking
            if self.cost_tracker:
                self.cost_tracker.start_analysis(run_id)
            
            # Fetch upfront data
            self.logger.info(f"Fetching data for {domain} analysis")
            data_result = await self.data_fetcher.fetch_domain_data(
                domain=domain,
                company=company,
                peer_list=peers or [],
                time_period=time_period
            )
            
            # Create analysis input
            analysis_input = AnalysisInput(
                run_id=run_id,
                company=company,
                ticker=company,
                domain=domain,
                time_period=time_period,
                supplied_metrics=data_result.financial_metrics,
                supplied_ratios=data_result.financial_ratios,
                peer_comparison=data_result.peer_comparison,
                user_focus=user_focus,
                persona=persona,
                max_tool_calls=6,
                max_loops=max_loops or DEFAULT_MAX_LOOPS
            )
            
            # Create domain agent
            domain_agent = self.agent_factory.create_domain_agent(domain)
            
            # Run analysis
            self.logger.info(f"Running {domain} analysis")
            analysis_output, review_history, run_cost = await domain_agent.analyze(analysis_input)

            # Enrich with ratio definitions (if ResearchReportOutput)
            if isinstance(analysis_output, ResearchReportOutput):
                self.logger.info("Enriching report with ratio definitions")
                try:
                    analysis_output = enrich_report_with_ratio_definitions(
                        analysis_output,
                        enable_debug=self.enable_debug
                    )
                    self.logger.info("Successfully enriched report with ratio definitions")
                except Exception as e:
                    self.logger.warning(f"Failed to enrich with ratio definitions: {e}")
                    # Continue with un-enriched report

            # Track execution
            execution_result = {
                "run_id": run_id,
                "type": "single_domain",
                "company": company,
                "domain": domain,
                "success": True,
                "analysis_output": analysis_output,
                "review_history": review_history,
                "data_quality": data_result.data_quality_score,
                "loops_used": len(review_history),
                "final_confidence": analysis_output.confidence_level,
                "execution_time": datetime.now().isoformat(),
                "run_cost": run_cost.get_breakdown(),
                "cost_breakdown": format_cost_summary(run_cost),
                "total_cost": run_cost.total_cost
            }
            
            self.execution_history.append(execution_result)
            self.active_analyses[run_id] = {
                "domain_agent": domain_agent,
                "analysis_output": analysis_output,
                "review_history": review_history
            }
            
            # Display cost summary
            self.logger.info("\n" + format_cost_summary(run_cost))

            # Update master agent memory
            await self.master_agent.arun(f"""
            Single domain analysis completed:
            Company: {company}
            Domain: {domain}
            Confidence: {analysis_output.confidence_level}
            Loops used: {len(review_history)}
            Quality score: {review_history[-1].quality_score if review_history else 'N/A'}
            Total cost: ${run_cost.total_cost:.4f}
            """)

            self.logger.info(f"Single domain analysis completed: {run_id}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Single domain analysis failed: {e}")

            # Try to store domain_agent in active_analyses even on failure
            # This allows chat session to start even if analysis failed (useful for debugging)
            try:
                if 'domain_agent' in locals():
                    self.active_analyses[run_id] = {
                        "domain_agent": domain_agent,
                        "analysis_output": None,  # No analysis output on failure
                        "review_history": [],
                        "success": False,
                        "error": str(e)
                    }
                    self.logger.info(f"Stored failed analysis in active_analyses for debugging")
            except:
                pass

            # Record failure
            failure_result = {
                "run_id": run_id,
                "type": "single_domain",
                "company": company,
                "domain": domain,
                "success": False,
                "error": str(e),
                "execution_time": datetime.now().isoformat()
            }

            self.execution_history.append(failure_result)
            return failure_result
    
    async def run_multi_domain_analysis(self,
                                       request: MasterAnalysisRequest) -> Dict[str, Any]:
        """
        Run parallel analysis across multiple domains.
        
        Args:
            request: Master analysis request with domains and parameters
            
        Returns:
            Comprehensive multi-domain analysis results
        """
        run_id = f"master_{request.company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting multi-domain analysis: {run_id}")
        self.logger.info(f"Domains: {request.domains}")
        
        # Store in master agent memory
        await self.master_agent.arun(f"""
        Starting comprehensive multi-domain analysis:
        Run ID: {run_id}
        Company: {request.company}
        Domains: {', '.join(request.domains)}
        Parallel execution: {request.run_parallel}
        Focus: {request.user_focus}
        """)
        
        try:
            # Validate domains
            invalid_domains = [d for d in request.domains if d not in get_available_domains()]
            if invalid_domains:
                raise ValueError(f"Unsupported domains: {invalid_domains}")
            
            # Initialize cost tracking
            if self.cost_tracker:
                self.cost_tracker.start_analysis(run_id)
            
            # Determine execution strategy
            if request.run_parallel:
                domain_results = await self._run_parallel_domains(request, run_id)
            else:
                domain_results = await self._run_sequential_domains(request, run_id)
            
            # Synthesize results
            master_synthesis = await self._synthesize_multi_domain_results(
                domain_results, request, run_id
            )
            
            # Create comprehensive result
            execution_result = {
                "run_id": run_id,
                "type": "multi_domain",
                "company": request.company,
                "domains": request.domains,
                "success": True,
                "domain_results": domain_results,
                "master_synthesis": master_synthesis,
                "execution_strategy": "parallel" if request.run_parallel else "sequential",
                "total_domains": len(request.domains),
                "successful_domains": len([r for r in domain_results.values() if r.get("success")]),
                "execution_time": datetime.now().isoformat(),
                "total_cost": self.cost_tracker.get_analysis_cost(run_id) if self.cost_tracker else 0.0
            }
            
            self.execution_history.append(execution_result)
            
            # Update master agent memory
            await self.master_agent.arun(f"""
            Multi-domain analysis completed:
            Company: {request.company}
            Domains completed: {execution_result['successful_domains']}/{execution_result['total_domains']}
            Strategy: {execution_result['execution_strategy']}
            Overall quality: {master_synthesis.get('overall_confidence', 'unknown')}
            """)
            
            self.logger.info(f"Multi-domain analysis completed: {run_id}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Multi-domain analysis failed: {e}")
            
            failure_result = {
                "run_id": run_id,
                "type": "multi_domain",
                "company": request.company,
                "domains": request.domains,
                "success": False,
                "error": str(e),
                "execution_time": datetime.now().isoformat()
            }
            
            self.execution_history.append(failure_result)
            return failure_result
    
    async def _run_parallel_domains(self, 
                                   request: MasterAnalysisRequest, 
                                   run_id: str) -> Dict[str, Dict[str, Any]]:
        """Run domain analyses in parallel."""
        
        self.logger.info(f"Running {len(request.domains)} domains in parallel")
        
        # Create tasks for parallel execution
        tasks = []
        for domain in request.domains:
            task = self.run_single_domain_analysis(
                company=request.company,
                domain=domain,
                peers=[],  # Simplified for parallel execution
                user_focus=request.user_focus,
                time_period=request.time_period,
                max_loops=2  # Reduced for parallel efficiency
            )
            tasks.append((domain, task))
        
        # Execute in parallel
        results = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks],
            return_exceptions=True
        )
        
        # Process results
        for i, (domain, _) in enumerate(tasks):
            if isinstance(completed_tasks[i], Exception):
                results[domain] = {
                    "success": False,
                    "error": str(completed_tasks[i]),
                    "domain": domain
                }
            else:
                results[domain] = completed_tasks[i]
        
        return results
    
    async def _run_sequential_domains(self, 
                                     request: MasterAnalysisRequest, 
                                     run_id: str) -> Dict[str, Dict[str, Any]]:
        """Run domain analyses sequentially."""
        
        self.logger.info(f"Running {len(request.domains)} domains sequentially")
        
        results = {}
        
        for domain in request.domains:
            try:
                result = await self.run_single_domain_analysis(
                    company=request.company,
                    domain=domain,
                    peers=[],
                    user_focus=request.user_focus,
                    time_period=request.time_period
                )
                results[domain] = result
                
            except Exception as e:
                self.logger.error(f"Sequential domain {domain} failed: {e}")
                results[domain] = {
                    "success": False,
                    "error": str(e),
                    "domain": domain
                }
        
        return results
    
    async def _synthesize_multi_domain_results(self,
                                              domain_results: Dict[str, Dict[str, Any]],
                                              request: MasterAnalysisRequest,
                                              run_id: str) -> Dict[str, Any]:
        """Synthesize multi-domain results into unified insights."""
        
        successful_results = {k: v for k, v in domain_results.items() if v.get("success")}
        
        # Create synthesis prompt
        synthesis_prompt = f"""
        Synthesize comprehensive financial analysis for {request.company}:
        
        COMPLETED DOMAINS: {', '.join(successful_results.keys())}
        FOCUS: {request.user_focus}
        
        KEY FINDINGS PER DOMAIN:
        """
        
        for domain, result in successful_results.items():
            if "analysis_output" in result:
                analysis = result["analysis_output"]
                synthesis_prompt += f"""
        
        {domain.upper()}:
        - Confidence: {analysis.confidence_level}
        - Key findings: {analysis.key_findings[:2]}
        """
        
        synthesis_prompt += """
        
        Provide unified executive synthesis:
        1. Overall financial health assessment
        2. Cross-domain insights and patterns
        3. Integrated risk assessment
        4. Strategic recommendations
        5. Investment thesis summary
        """
        
        # Run synthesis through master agent
        synthesis_response = await self.master_agent.arun(synthesis_prompt)
        
        return {
            "synthesis_text": str(synthesis_response),
            "domains_analyzed": list(successful_results.keys()),
            "overall_confidence": self._calculate_overall_confidence(successful_results),
            "cross_domain_insights": self._extract_cross_domain_insights(successful_results),
            "unified_recommendations": self._extract_unified_recommendations(successful_results)
        }
    
    def _calculate_overall_confidence(self, successful_results: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall confidence across domains."""
        
        confidence_scores = {"high": 3, "medium": 2, "low": 1}
        
        total_score = 0
        count = 0
        
        for result in successful_results.values():
            if "analysis_output" in result:
                confidence = result["analysis_output"].confidence_level
                total_score += confidence_scores.get(confidence, 1)
                count += 1
        
        if count == 0:
            return "low"
        
        avg_score = total_score / count
        
        if avg_score >= 2.5:
            return "high"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "low"
    
    def _extract_cross_domain_insights(self, successful_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Extract insights that span multiple domains."""
        
        # Simplified cross-domain insight extraction
        insights = []
        
        if len(successful_results) >= 2:
            insights.append("Multi-domain analysis provides comprehensive financial perspective")
        
        if "liquidity" in successful_results and "leverage" in successful_results:
            insights.append("Liquidity and leverage profiles assessed for balanced risk evaluation")
        
        return insights
    
    def _extract_unified_recommendations(self, successful_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Extract unified recommendations across domains."""
        
        all_recommendations = []
        
        for result in successful_results.values():
            if "analysis_output" in result:
                all_recommendations.extend(result["analysis_output"].recommendations)
        
        # Return unique recommendations (simplified)
        return list(set(all_recommendations))[:5]
    
    async def start_chat_session(self, run_id: str) -> Dict[str, Any]:
        """Start interactive chat session for a completed analysis."""

        if run_id not in self.active_analyses:
            return {"error": "Analysis not found or not active"}

        analysis_data = self.active_analyses[run_id]

        # Check if analysis failed
        if not analysis_data.get("success", True):
            return {
                "error": f"Cannot start chat session for failed analysis. Error: {analysis_data.get('error', 'Unknown')}",
                "suggestion": "Please run a successful analysis first before starting a chat session."
            }

        # Check if analysis_output exists
        if not analysis_data.get("analysis_output"):
            return {
                "error": "No analysis output available for chat session",
                "suggestion": "The analysis may have failed. Please run the analysis again."
            }

        session_id = f"chat_{run_id}_{datetime.now().strftime('%H%M%S')}"

        return await self.chat_interface.start_chat_session(
            session_id=session_id,
            domain_agent=analysis_data["domain_agent"],
            analysis_output=analysis_data["analysis_output"],
            review_history=analysis_data["review_history"]
        )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all executions."""

        total_analyses = len(self.execution_history)
        successful_analyses = len([e for e in self.execution_history if e.get("success")])

        return {
            "total_analyses": total_analyses,
            "successful_analyses": successful_analyses,
            "success_rate": successful_analyses / total_analyses if total_analyses > 0 else 0,
            "active_analyses": len(self.active_analyses),
            "available_domains": get_available_domains(),
            "cost_tracking_enabled": ENABLE_COST_TRACKING,
            "last_execution": self.execution_history[-1]["execution_time"] if self.execution_history else None
        }

    def format_research_report(self, report_output: Any) -> str:
        """
        Format ResearchReportOutput as a human-readable professional research report.

        Args:
            report_output: ResearchReportOutput object

        Returns:
            Formatted report string ready for display
        """
        # Import here to avoid circular dependencies
        from config.schemas import ResearchReportOutput

        # Format research-grade report
        report_lines = []

        # Header
        report_lines.append("═" * 80)
        report_lines.append(f"  FINANCIAL ANALYSIS RESEARCH REPORT")
        report_lines.append("═" * 80)
        report_lines.append(f"Company: {report_output.company} ({report_output.ticker})")
        report_lines.append(f"Domain: {report_output.domain.upper()}")
        report_lines.append(f"Date: {report_output.analysis_timestamp[:10]}")
        report_lines.append(f"Confidence: {report_output.confidence_level.upper()}")
        report_lines.append("═" * 80)
        report_lines.append("")

        # Executive Summary
        report_lines.append("📊 EXECUTIVE SUMMARY")
        report_lines.append("─" * 80)
        report_lines.append(report_output.executive_summary)
        report_lines.append("")

        # Chart-Ready Ratios
        if report_output.chartable_ratios:
            report_lines.append("📈 KEY RATIOS (Chart-Ready)")
            report_lines.append("─" * 80)
            for i, ratio in enumerate(report_output.chartable_ratios, 1):
                report_lines.append(f"\n{i}. {ratio.ratio_name}")
                report_lines.append(f"   Trend: {ratio.trend_direction.upper()}")
                report_lines.append(f"   Interpretation: {ratio.interpretation}")

                # Company values
                company_vals = ", ".join([f"{q}: {v:.3f}" for q, v in sorted(ratio.company_values.items(), reverse=True)[:4]])
                report_lines.append(f"   Company: {company_vals}")

                # Peer comparison
                if ratio.peer_values:
                    for peer, peer_data in list(ratio.peer_values.items())[:3]:  # Show top 3 peers
                        peer_vals = ", ".join([f"{q}: {v:.3f}" for q, v in sorted(peer_data.items(), reverse=True)[:4]])
                        report_lines.append(f"   {peer}: {peer_vals}")
            report_lines.append("")

        # Chart-Ready Metrics
        if report_output.chartable_metrics:
            report_lines.append("📊 KEY METRICS (Chart-Ready)")
            report_lines.append("─" * 80)
            for i, metric in enumerate(report_output.chartable_metrics, 1):
                report_lines.append(f"\n{i}. {metric.metric_name}")
                report_lines.append(f"   Trend: {metric.trend_direction.upper()}")
                report_lines.append(f"   Unit: {metric.unit}")
                report_lines.append(f"   Interpretation: {metric.interpretation}")

                # Values over time
                metric_vals = ", ".join([f"{q}: {self._format_metric_value(v, metric.unit)}"
                                        for q, v in sorted(metric.values.items(), reverse=True)[:6]])
                report_lines.append(f"   Values: {metric_vals}")
            report_lines.append("")

        # Analysis Bullets
        if report_output.analysis_bullets:
            report_lines.append("🔍 DETAILED ANALYSIS")
            report_lines.append("─" * 80)
            for i, bullet in enumerate(report_output.analysis_bullets, 1):
                # Bullet text with importance
                importance_emoji = {"critical": "🔴", "high": "🟠", "moderate": "🟡", "low": "🟢"}
                emoji = importance_emoji.get(bullet.importance, "⚪")
                report_lines.append(f"\n{i}. {emoji} {bullet.bullet_text}")

                # Quantitative evidence
                if bullet.quantitative_evidence:
                    report_lines.append("   📊 Quantitative Evidence:")
                    for evidence in bullet.quantitative_evidence:
                        report_lines.append(f"      • {evidence}")

                # Qualitative sources
                if bullet.qualitative_sources:
                    report_lines.append("   📚 Sources:")
                    for source in bullet.qualitative_sources:
                        source_dict = source if isinstance(source, dict) else source.to_dict() if hasattr(source, 'to_dict') else {}
                        source_parts = [source_dict.get('tool_name', 'Unknown')]
                        if source_dict.get('period'):
                            source_parts.append(source_dict['period'])
                        if source_dict.get('speaker_name'):
                            source_parts.append(f"Speaker: {source_dict['speaker_name']}")
                        if source_dict.get('report_type'):
                            source_parts.append(source_dict['report_type'])
                        report_lines.append(f"      • {', '.join(source_parts)}")
                        if source_dict.get('chunk_summary'):
                            report_lines.append(f"        └─ {source_dict['chunk_summary'][:80]}...")
            report_lines.append("")

        # Recommendations
        if report_output.recommendations:
            report_lines.append("💡 RECOMMENDATIONS")
            report_lines.append("─" * 80)
            for i, rec in enumerate(report_output.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")

        # Risk Factors
        if report_output.risk_factors:
            report_lines.append("⚠️  RISK FACTORS")
            report_lines.append("─" * 80)
            for i, risk in enumerate(report_output.risk_factors, 1):
                report_lines.append(f"{i}. {risk}")
            report_lines.append("")

        # Footer
        report_lines.append("═" * 80)
        report_lines.append(f"Data Sources: {report_output.data_sources_summary}")
        report_lines.append(f"Tool Calls: {len(report_output.tool_calls_made)}")
        report_lines.append(f"Report ID: {report_output.run_id}")
        report_lines.append("═" * 80)

        return "\n".join(report_lines)

    def _format_metric_value(self, value: float, unit: str) -> str:
        """Format metric value based on unit."""
        if "USD" in unit or "$" in unit:
            if abs(value) >= 1e9:
                return f"${value/1e9:.1f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.1f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.1f}K"
            else:
                return f"${value:.2f}"
        else:
            return f"{value:,.0f}"



class CostTracker:
    """Simple cost tracking for budget management."""
    
    def __init__(self):
        self.analysis_costs: Dict[str, float] = {}
        self.total_cost = 0.0
    
    def start_analysis(self, run_id: str):
        """Start cost tracking for analysis."""
        self.analysis_costs[run_id] = 0.0
    
    def add_cost(self, run_id: str, cost: float):
        """Add cost to analysis."""
        if run_id in self.analysis_costs:
            self.analysis_costs[run_id] += cost
            self.total_cost += cost
    
    def get_analysis_cost(self, run_id: str) -> float:
        """Get cost for specific analysis."""
        return self.analysis_costs.get(run_id, 0.0)
    
    def get_total_cost(self) -> float:
        """Get total accumulated cost."""
        return self.total_cost