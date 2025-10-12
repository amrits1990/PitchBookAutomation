"""
Domain Agent - Analyst-Reviewer Loop Implementation
"""

import logging
import json
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseFinancialAgent
from .analyst_agent import AnalystAgent
from .reviewer_agent import ReviewerAgent
from config.schemas import AnalysisInput, ResearchReportOutput, ReviewResult, ChatRefinementRequest
from config.settings import DEFAULT_MAX_LOOPS, CHAT_MAX_LOOPS, MAX_LOOPS_HARD_LIMIT
from config.llm_cost_tracker import get_global_tracker, AgentCost, RunCost


class DomainAgent:
    """
    Domain Agent implementing configurable analyst-reviewer loops.
    
    Flow:
    1. Analyst analyzes → Reviewer evaluates → Loop until approved or max loops
    2. Chat interface for post-analysis refinements
    """
    
    def __init__(self, domain: str, enable_debug: bool = False):
        """
        Initialize domain agent for specific domain.
        
        Args:
            domain: Domain type (liquidity, leverage, etc.)
            enable_debug: Enable debug logging
        """
        self.domain = domain
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(f"{__name__}.{domain}")
        
        if enable_debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Initialize analyst and reviewer agents
        self.analyst = AnalystAgent(domain=domain, enable_debug=enable_debug)
        self.reviewer = ReviewerAgent(domain=domain, enable_debug=enable_debug)
        
        # Loop tracking
        self.current_loops = 0
        self.max_loops = DEFAULT_MAX_LOOPS
        self.analysis_history = []
        self.cumulative_tool_calls = []  # Track ALL tool calls across loops

        self.logger.info(f"DomainAgent initialized for {domain}")
    
    async def analyze(self,
                     analysis_input: AnalysisInput,
                     max_loops: int = None) -> tuple[ResearchReportOutput, List[ReviewResult], RunCost]:
        """
        Run complete analyst-reviewer loop until approval or max loops.
        
        Args:
            analysis_input: Input data and parameters
            max_loops: Override default max loops
            
        Returns:
            Tuple of (final_analysis, review_history)
        """
        if max_loops is not None:
            self.max_loops = min(max_loops, MAX_LOOPS_HARD_LIMIT)
        else:
            self.max_loops = min(analysis_input.max_loops, MAX_LOOPS_HARD_LIMIT)
        
        self.logger.info(f"Starting analysis loop for {analysis_input.company} - {self.domain}")
        self.logger.info(f"Max loops configured: {self.max_loops}")

        # Initialize cost tracking
        tracker = get_global_tracker()
        run_cost = tracker.start_run(analysis_input.run_id, analysis_input.company, analysis_input.domain)

        self.current_loops = 0
        self.analysis_history = []
        self.cumulative_tool_calls = []  # Reset for new analysis
        review_history = []

        current_analysis = None
        
        # Main analyst-reviewer loop
        while self.current_loops < self.max_loops:
            self.current_loops += 1
            loop_start_time = datetime.now()
            
            self.logger.info(f"Starting loop {self.current_loops}/{self.max_loops}")
            
            # Step 1: Analyst Analysis
            try:
                # Include previous feedback if this is a revision loop
                previous_feedback = None
                if review_history:
                    previous_feedback = review_history[-1]

                # Create a copy of current_analysis with cumulative tool calls for history context
                analysis_with_history = None
                if current_analysis and self.cumulative_tool_calls:
                    # Clone the analysis and replace tool_calls_made with cumulative history
                    analysis_with_history = current_analysis
                    # Override tool_calls_made with cumulative history for the analyst to see
                    analysis_with_history.tool_calls_made = self.cumulative_tool_calls.copy()
                    self.logger.info(f"DEBUG_TRACE: Passing cumulative tool history to analyst: {len(self.cumulative_tool_calls)} total calls")

                current_analysis, analyst_token_usage = await self.analyst.analyze(
                    analysis_input=analysis_input,
                    previous_analysis=analysis_with_history if analysis_with_history else current_analysis,
                    previous_feedback=previous_feedback,
                    loop_iteration=self.current_loops
                )

                # Track analyst cost
                analyst_cost = AgentCost(
                    agent_name=f"{self.domain}_analyst",
                    agent_type="analyst",
                    loop_number=self.current_loops
                )
                analyst_cost.add_usage(analyst_token_usage)
                run_cost.add_analyst_cost(analyst_cost)

                # DEBUG: Check what the analyst actually returned
                self.logger.info(f"DEBUG_TRACE: Analyst returned analysis for {current_analysis.company}")
                bullets_count = len(getattr(current_analysis, 'analysis_bullets', [])) if getattr(current_analysis, 'analysis_bullets', None) else 0
                self.logger.info(f"DEBUG_TRACE: Analysis has {bullets_count} analysis bullets")
                self.logger.info(f"DEBUG_TRACE: Analysis has {len(current_analysis.tool_calls_made)} tool calls")
                analysis_bullets = getattr(current_analysis, 'analysis_bullets', [])
                qual_sources_count = sum(len(getattr(b, 'qualitative_sources', [])) for b in analysis_bullets) if analysis_bullets else 0
                self.logger.info(f"DEBUG_TRACE: Analysis has {qual_sources_count} qualitative sources")
                first_bullet = analysis_bullets[0].bullet_text[:100] if analysis_bullets else 'NONE'
                self.logger.info(f"DEBUG_TRACE: First analysis bullet: {first_bullet}")

                # Accumulate tool calls from this loop
                if current_analysis.tool_calls_made:
                    self.cumulative_tool_calls.extend(current_analysis.tool_calls_made)
                    self.logger.info(f"DEBUG_TRACE: Cumulative tool calls now: {len(self.cumulative_tool_calls)} (added {len(current_analysis.tool_calls_made)} from this loop)")

                self.analysis_history.append({
                    "loop": self.current_loops,
                    "analysis": current_analysis,
                    "timestamp": datetime.now().isoformat()
                })
                
                self.logger.info(f"Loop {self.current_loops}: Analysis completed")
                
            except Exception as e:
                self.logger.error(f"Loop {self.current_loops}: Analyst failed - {e}")
                # Return best available analysis with cost tracking
                run_cost.finalize()
                if current_analysis:
                    return current_analysis, review_history, run_cost
                else:
                    raise Exception(f"Analysis failed on loop {self.current_loops}: {e}")
            
            # Step 2: Reviewer Evaluation  
            try:
                # DEBUG: Check what we're passing to the reviewer
                bullets_count = len(getattr(current_analysis, 'analysis_bullets', [])) if getattr(current_analysis, 'analysis_bullets', None) else 0
                self.logger.info(f"DEBUG_TRACE: About to pass to reviewer - analysis has {bullets_count} analysis bullets")
                self.logger.info(f"DEBUG_TRACE: About to pass to reviewer - analysis has {len(current_analysis.tool_calls_made)} tool calls")
                
                review_result, reviewer_token_usage = await self.reviewer.evaluate(
                    analysis=current_analysis,
                    input_data=analysis_input,
                    loop_iteration=self.current_loops,
                    max_loops=self.max_loops
                )

                # Track reviewer cost
                reviewer_cost = AgentCost(
                    agent_name=f"{self.domain}_reviewer",
                    agent_type="reviewer",
                    loop_number=self.current_loops
                )
                reviewer_cost.add_usage(reviewer_token_usage)
                run_cost.add_reviewer_cost(reviewer_cost)

                review_history.append(review_result)
                
                self.logger.info(f"Loop {self.current_loops}: Review completed - {review_result.verdict}")
                self.logger.info(f"Quality score: {review_result.quality_score:.2f}")
                
            except Exception as e:
                self.logger.error(f"Loop {self.current_loops}: Reviewer failed - {e}")
                # If reviewer fails but we have analysis, continue
                if current_analysis:
                    # Create default "approved" review to end loop
                    default_review = ReviewResult(
                        verdict="approved",
                        quality_score=0.7,
                        strengths=["Analysis completed despite review system issue"],
                        weaknesses=["Unable to complete automated review"],
                        specific_improvements=[],
                        should_continue=False,
                        reason="Reviewer system unavailable"
                    )
                    review_history.append(default_review)
                    break
                else:
                    raise Exception(f"Both analysis and review failed on loop {self.current_loops}")
            
            # Step 3: Loop Decision
            # DEBUG: Log review result details
            self.logger.info(f"DEBUG_TRACE: Review result - verdict: {review_result.verdict}, should_continue: {review_result.should_continue}, score: {review_result.quality_score}")
            
            if review_result.verdict == "approved":
                self.logger.info(f"Analysis approved after {self.current_loops} loops")
                break
            elif self.current_loops >= self.max_loops:
                self.logger.warning(f"Maximum loops ({self.max_loops}) reached with verdict: {review_result.verdict}")
                break
            elif review_result.verdict == "rejected" and not review_result.should_continue:
                self.logger.warning(f"Analysis rejected with no continuation after {self.current_loops} loops (should_continue={review_result.should_continue})")
                break
            elif not review_result.should_continue:
                self.logger.info(f"Review indicates no further improvement possible after {self.current_loops} loops (should_continue={review_result.should_continue})")
                break
            else:
                # Continue to next loop with feedback (including rejected analyses that can be improved)
                self.logger.info(f"Continuing to loop {self.current_loops + 1} with feedback (verdict: {review_result.verdict}, should_continue: {review_result.should_continue})")
                loop_duration = (datetime.now() - loop_start_time).total_seconds()
                self.logger.debug(f"Loop {self.current_loops} duration: {loop_duration:.1f} seconds")
        
        # Final result
        total_duration = sum([(datetime.fromisoformat(h["timestamp"]) - datetime.fromisoformat(self.analysis_history[0]["timestamp"])).total_seconds()
                             for h in self.analysis_history[1:]], 0)

        # Finalize cost tracking
        run_cost.finalize()

        self.logger.info(f"Analysis completed: {self.current_loops} loops, {total_duration:.1f} seconds")
        self.logger.info(f"Total cost for this run: ${run_cost.total_cost:.4f}")

        return current_analysis, review_history, run_cost
    
    async def chat_refinement(self,
                            refinement_request: ChatRefinementRequest) -> tuple[ResearchReportOutput, List[ReviewResult], RunCost]:
        """
        Handle user chat refinement requests with reduced loop count.
        
        Args:
            refinement_request: User's refinement request
            
        Returns:
            Tuple of (refined_analysis, review_history)
        """
        self.logger.info(f"Processing chat refinement: {refinement_request.refinement_type}")
        
        # Create modified input for refinement
        chat_input = AnalysisInput(
            run_id=f"{refinement_request.original_analysis.run_id}_chat_{datetime.now().strftime('%H%M%S')}",
            company=refinement_request.original_analysis.company,
            ticker=refinement_request.original_analysis.company,  # Simplified
            domain=refinement_request.original_analysis.domain,
            time_period="chat_refinement",
            supplied_metrics={},  # Will be filled by analyst if needed
            supplied_ratios={},
            peer_comparison={},
            user_focus=refinement_request.user_question,
            max_tool_calls=3,  # Reduced for chat
            max_loops=min(refinement_request.max_loops, CHAT_MAX_LOOPS)
        )
        
        # Run refinement analysis with reduced loops
        refined_analysis, review_history, run_cost = await self.analyze(
            analysis_input=chat_input,
            max_loops=chat_input.max_loops
        )

        self.logger.info(f"Chat refinement completed with {len(review_history)} review cycles")

        return refined_analysis, review_history, run_cost
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis process."""
        if not self.analysis_history:
            return {"status": "no_analysis"}
        
        latest_analysis = self.analysis_history[-1]["analysis"]

        bullets_count = len(getattr(latest_analysis, 'analysis_bullets', [])) if getattr(latest_analysis, 'analysis_bullets', None) else 0
        return {
            "domain": self.domain,
            "company": latest_analysis.company,
            "total_loops": self.current_loops,
            "final_confidence": latest_analysis.confidence_level,
            "analysis_bullets_count": bullets_count,
            "tool_calls_made": len(latest_analysis.tool_calls_made),
            "analysis_duration": len(self.analysis_history),
            "last_updated": self.analysis_history[-1]["timestamp"]
        }
    
    def get_loop_history(self) -> List[Dict[str, Any]]:
        """Get detailed loop history for debugging."""
        return [
            {
                "loop": h["loop"],
                "timestamp": h["timestamp"],
                "confidence": h["analysis"].confidence_level,
                "bullets_count": len(getattr(h["analysis"], 'analysis_bullets', [])) if getattr(h["analysis"], 'analysis_bullets', None) else 0,
                "summary_length": len(h["analysis"].executive_summary)
            }
            for h in self.analysis_history
        ]