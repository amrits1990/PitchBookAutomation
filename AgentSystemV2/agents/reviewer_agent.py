"""
Reviewer Agent - Evaluates analysis quality and provides feedback
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

from .base_agent import BaseFinancialAgent
from config.schemas import AnalysisInput, ResearchReportOutput, ReviewResult, create_quality_evaluation_prompt
from config.domain_configs import get_domain_config
from config.settings import REVIEWER_MODEL_ID, QUALITY_THRESHOLD
from config.llm_cost_tracker import TokenUsage


class ReviewerAgent(BaseFinancialAgent):
    """
    Reviewer agent that evaluates analysis quality and provides feedback.
    Determines whether analysis should be approved, revised, or rejected.
    """
    
    def __init__(self, domain: str, enable_debug: bool = False):
        """
        Initialize reviewer for specific domain.
        
        Args:
            domain: Domain type (liquidity, leverage, etc.)
            enable_debug: Enable debug logging
        """
        self.domain = domain
        self.domain_config = get_domain_config(domain)
        
        super().__init__(
            agent_name=f"{domain}_reviewer",
            model_id=REVIEWER_MODEL_ID,
            enable_debug=enable_debug,
            use_memory=False  # Stateless for cost optimization
        )
    
    def _get_agent_instructions(self) -> str:
        """Get reviewer agent instructions."""
        return f"""
You are a {self.domain} analysis reviewer specializing in quality assessment.

═══════════════════════════════════════════════════════════════
                    🎯 YOUR PRIMARY MISSION
═══════════════════════════════════════════════════════════════

Judge whether the analysis actually ANSWERS THE USER'S QUESTION with sufficient depth and insight.

DO NOT just check format compliance. Focus on:
1. Does it address what the user asked for?
2. Is the analysis deep and insightful, or superficial?
3. Can the target audience make decisions from this?
4. Are the insights valuable and non-obvious?

═══════════════════════════════════════════════════════════════
                    ⚖️ EVALUATION BALANCE (50/50)
═══════════════════════════════════════════════════════════════

A. CONTENT QUALITY (50% of your evaluation):
   - Relevance to user's specific query
   - Depth of analysis (WHY things happened, not just WHAT)
   - Actionability for the target persona
   - Completeness of coverage
   - Quality of insights (non-obvious, valuable)

B. TECHNICAL QUALITY (50% of your evaluation):
   - Evidence backing (data from supplied metrics)
   - Logical consistency
   - Data discipline (no numbers from tool outputs)
   - Source attribution for qualitative claims
   - {chr(10).join([f"   - {criterion}" for criterion in self.domain_config.reviewer_criteria])}

═══════════════════════════════════════════════════════════════
                    ⚠️ COMMON PROBLEMS TO CATCH
═══════════════════════════════════════════════════════════════

CONTENT ISSUES:
- Analysis doesn't actually answer the user's question
- Superficial ("cash increased") vs deep ("cash increased 15% QoQ due to working capital efficiency, with DPO improving from 45 to 52 days")
- Generic insights that could apply to any company
- Missing key aspects the user asked about
- Recommendations too vague to act on

TECHNICAL ISSUES:
- Numbers extracted from transcripts instead of supplied_metrics
- Claims without evidence
- Missing source attribution
- Incomplete peer comparisons user requested
- Peer comparison errors: Mixing different temporal periods (e.g., comparing Period 1 AAPL with Period 2 MSFT)
- Risks not comprehensive

═══════════════════════════════════════════════════════════════
                    📊 VERDICT GUIDELINES
═══════════════════════════════════════════════════════════════

"approved":
- Answers user query thoroughly ✓
- Deep, insightful analysis ✓
- Technically sound ✓
- Score >= 0.8

"needs_revision":
- Partially answers query or lacks depth
- Has potential but needs improvement
- Score 0.5-0.8
- Improvements are achievable

"rejected":
- Misses user's query entirely
- Superficial or fundamentally flawed
- Score < 0.5

═══════════════════════════════════════════════════════════════
                    💬 FEEDBACK APPROACH
═══════════════════════════════════════════════════════════════

- Be SPECIFIC: Not "needs more depth", but "Explain WHY cash increased, not just that it did"
- Reference USER QUERY: "User asked about working capital but analysis doesn't cover accounts payable trends"
- Be CONSTRUCTIVE: Suggest what tool calls or analysis would fix the issue
- Consider LOOP CONTEXT: Be stricter on loop 1, more lenient on final loop

OUTPUT FORMAT: ReviewResult schema as valid JSON.
"""
    
    async def evaluate(self,
                      analysis: ResearchReportOutput,
                      input_data: AnalysisInput,
                      loop_iteration: int = 1,
                      max_loops: int = 3) -> tuple[ReviewResult, TokenUsage]:
        """
        Evaluate analysis quality and provide feedback.
        
        Args:
            analysis: Analysis output to evaluate
            input_data: Original analysis input for context
            loop_iteration: Current loop number
            max_loops: Maximum allowed loops
            
        Returns:
            Review result with verdict and feedback
        """
        self.logger.info(f"Evaluating {self.domain} analysis (loop {loop_iteration}/{max_loops})")
        
        # DEBUG: Check what the reviewer received
        self.logger.info(f"DEBUG_TRACE: Reviewer received analysis for {analysis.company}")
        bullets_count = len(analysis.analysis_bullets) if analysis.analysis_bullets else 0
        self.logger.info(f"DEBUG_TRACE: Reviewer sees {bullets_count} analysis bullets")
        self.logger.info(f"DEBUG_TRACE: Reviewer sees {len(analysis.tool_calls_made)} tool calls")
        qual_sources_count = sum(len(b.qualitative_sources) for b in analysis.analysis_bullets) if analysis.analysis_bullets else 0
        self.logger.info(f"DEBUG_TRACE: Reviewer sees {qual_sources_count} qualitative sources")
        if analysis.analysis_bullets:
            self.logger.info(f"DEBUG_TRACE: First analysis bullet: {analysis.analysis_bullets[0].bullet_text[:100]}")
        
        # Build evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(
            analysis=analysis,
            input_data=input_data,
            loop_iteration=loop_iteration,
            max_loops=max_loops
        )
        
        # Add context
        context = {
            "domain": self.domain,
            "company": analysis.company,
            "loop_iteration": loop_iteration,
            "max_loops": max_loops
        }
        
        try:
            # Run evaluation - now returns (response, tool_calls, token_usage)
            response, _, token_usage = await self._run_agent(evaluation_prompt, context)

            # Parse response into ReviewResult
            review_result = self._parse_review_response(
                response=response,
                loop_iteration=loop_iteration,
                max_loops=max_loops
            )

            self.logger.info(f"Review completed - verdict: {review_result.verdict}, score: {review_result.quality_score:.2f}")
            return review_result, token_usage

        except Exception as e:
            self.logger.error(f"Review failed: {e}")
            return self._create_fallback_review(loop_iteration, max_loops, str(e)), TokenUsage(model_id=self.model_id)
    
    def _build_evaluation_prompt(self,
                                analysis: ResearchReportOutput,
                                input_data: AnalysisInput,
                                loop_iteration: int,
                                max_loops: int) -> str:
        """Build comprehensive evaluation prompt."""
        
        # Basic evaluation prompt
        base_prompt = create_quality_evaluation_prompt(analysis, input_data)
        
        # Add domain-specific criteria
        criteria_section = f"""
DOMAIN-SPECIFIC EVALUATION CRITERIA:
{chr(10).join([f"{i+1}. {criterion}" for i, criterion in enumerate(self.domain_config.reviewer_criteria)])}
"""
        
        # Add loop context
        loop_context = f"""
LOOP CONTEXT:
- Current loop: {loop_iteration}/{max_loops}
- Time for improvement: {'Yes' if loop_iteration < max_loops else 'No'}
- Evaluation strategy: {'Strict' if loop_iteration == 1 else 'Progressive'}
"""
        
        # Quality scoring guidelines
        scoring_guidelines = f"""
QUALITY SCORING GUIDELINES (0.0 - 1.0):
- 0.9-1.0: Exceptional analysis, publication ready
- 0.8-0.9: High quality, minor improvements possible
- 0.7-0.8: Good analysis, some notable improvements needed
- 0.6-0.7: Adequate analysis, significant improvements required
- 0.5-0.6: Below standard, major revision needed
- <0.5: Poor quality, fundamental issues present

VERDICT GUIDELINES:
- "approved": Score >= {QUALITY_THRESHOLD} OR last loop with score >= 0.6
- "needs_revision": Score < {QUALITY_THRESHOLD} AND improvements possible AND loops remaining
- "rejected": Score < 0.5 AND fundamental issues OR last loop with score < 0.6
"""
        
        # Combine sections
        full_prompt = f"""
{base_prompt}

{criteria_section}

{loop_context}

{scoring_guidelines}

EVALUATION STEPS:
1. CHECK RELEVANCE: Does it answer the user's query? ({input_data.user_focus})
2. ASSESS DEPTH: Is analysis deep (explains WHY) or superficial (just describes WHAT)?
3. VERIFY ACTIONABILITY: Can {input_data.persona} make decisions from this?
4. CHECK COMPLETENESS: Are all requested aspects covered (peer comparison, time period, etc.)?
5. VALIDATE PEER COMPARISONS: If peer comparison included, did analyst compare only within same temporal periods?
6. EVALUATE TECHNICAL QUALITY: Evidence, logic, data discipline, sources
7. CALCULATE SCORE: Balance content quality (50%) + technical quality (50%)
8. PROVIDE FEEDBACK: Specific strengths, weaknesses, actionable improvements

⚠️ REMEMBER: Content quality = Technical quality in your scoring!
   Don't approve perfectly formatted but shallow analysis.

Return evaluation in ReviewResult format as valid JSON.
"""
        
        return full_prompt
    
    def _parse_review_response(self,
                              response: str,
                              loop_iteration: int,
                              max_loops: int) -> ReviewResult:
        """Parse review response into ReviewResult structure."""
        
        try:
            # Try to parse as JSON first
            if response.strip().startswith('{'):
                parsed = json.loads(response)
                return self._create_review_result_from_dict(parsed, loop_iteration, max_loops)
            
            # If not JSON, extract structured information
            return self._extract_review_from_text(response, loop_iteration, max_loops)
            
        except Exception as e:
            self.logger.warning(f"Failed to parse review response: {e}")
            return self._create_fallback_review(loop_iteration, max_loops, f"Parse error: {e}")
    
    def _create_review_result_from_dict(self,
                                       data: Dict[str, Any],
                                       loop_iteration: int,
                                       max_loops: int) -> ReviewResult:
        """Create ReviewResult from parsed dictionary."""
        
        verdict = data.get('verdict', 'needs_revision')
        quality_score = float(data.get('quality_score', 0.5))
        
        # Determine should_continue logic
        should_continue = self._determine_should_continue(
            verdict=verdict,
            quality_score=quality_score,
            loop_iteration=loop_iteration,
            max_loops=max_loops
        )
        
        return ReviewResult(
            verdict=verdict,
            quality_score=quality_score,
            strengths=data.get('strengths', []),
            weaknesses=data.get('weaknesses', []),
            specific_improvements=data.get('specific_improvements', []),
            should_continue=should_continue,
            reason=data.get('reason', f"Loop {loop_iteration}/{max_loops} evaluation")
        )
    
    def _extract_review_from_text(self,
                                 response: str,
                                 loop_iteration: int,
                                 max_loops: int) -> ReviewResult:
        """Extract review components from unstructured text."""
        
        # Simple extraction logic
        lines = response.split('\n')
        
        verdict = "needs_revision"
        quality_score = 0.5
        strengths = []
        weaknesses = []
        specific_improvements = []
        
        for line in lines:
            line = line.strip().lower()
            if not line:
                continue
            
            # Extract verdict
            if "approved" in line and "verdict" in line:
                verdict = "approved"
            elif "rejected" in line and "verdict" in line:
                verdict = "rejected"
            
            # Extract quality score
            if "score" in line or "quality" in line:
                import re
                score_match = re.search(r'(\d*\.?\d+)', line)
                if score_match:
                    try:
                        score = float(score_match.group(1))
                        if 0.0 <= score <= 1.0:
                            quality_score = score
                        elif 0 <= score <= 10:  # Scale conversion
                            quality_score = score / 10.0
                    except ValueError:
                        pass
            
            # Extract feedback lists (simplified)
            if "strength" in line and line.startswith(('-', '•', '*')):
                strengths.append(line[1:].strip())
            elif "weakness" in line and line.startswith(('-', '•', '*')):
                weaknesses.append(line[1:].strip())
            elif "improvement" in line and line.startswith(('-', '•', '*')):
                specific_improvements.append(line[1:].strip())
        
        should_continue = self._determine_should_continue(
            verdict=verdict,
            quality_score=quality_score,
            loop_iteration=loop_iteration,
            max_loops=max_loops
        )
        
        return ReviewResult(
            verdict=verdict,
            quality_score=quality_score,
            strengths=strengths[:3],  # Limit to 3
            weaknesses=weaknesses[:3],
            specific_improvements=specific_improvements[:3],
            should_continue=should_continue,
            reason=f"Text extraction from loop {loop_iteration}/{max_loops}"
        )
    
    def _determine_should_continue(self,
                                  verdict: str,
                                  quality_score: float,
                                  loop_iteration: int,
                                  max_loops: int) -> bool:
        """Determine if loop should continue based on verdict and context."""
        
        # DEBUG: Log the decision logic
        self.logger.info(f"DEBUG_TRACE: Determining should_continue for verdict='{verdict}', score={quality_score}, loop={loop_iteration}/{max_loops}")
        
        # Never continue if approved
        if verdict == "approved":
            self.logger.info(f"DEBUG_TRACE: Not continuing - analysis approved")
            return False
        
        # Never continue if at max loops
        if loop_iteration >= max_loops:
            self.logger.info(f"DEBUG_TRACE: Not continuing - at max loops")
            return False
        
        # For rejected analyses, continue if quality score suggests improvement is possible
        if verdict == "rejected":
            # Don't continue if quality is extremely low (< 0.3) - fundamental issues
            if quality_score < 0.3:
                self.logger.info(f"DEBUG_TRACE: Not continuing - quality too low ({quality_score} < 0.3)")
                return False
            # Continue if we have loops remaining and quality is not hopeless
            should_continue = loop_iteration < max_loops
            self.logger.info(f"DEBUG_TRACE: Rejected analysis - continuing: {should_continue}")
            return should_continue
        
        # For needs_revision, continue if quality is below threshold and we have loops remaining
        if verdict == "needs_revision":
            should_continue = quality_score < QUALITY_THRESHOLD and loop_iteration < max_loops
            self.logger.info(f"DEBUG_TRACE: Needs revision - continuing: {should_continue} (score {quality_score} < {QUALITY_THRESHOLD})")
            return should_continue
        
        # Default: don't continue for unknown verdicts
        self.logger.info(f"DEBUG_TRACE: Not continuing - unknown verdict '{verdict}'")
        return False
    
    def _create_fallback_review(self,
                               loop_iteration: int,
                               max_loops: int,
                               error_msg: str) -> ReviewResult:
        """Create fallback review when evaluation fails."""
        
        # Determine verdict based on loop position
        if loop_iteration >= max_loops:
            verdict = "approved"  # Accept whatever we have
        else:
            verdict = "needs_revision"
        
        return ReviewResult(
            verdict=verdict,
            quality_score=0.6,  # Neutral score
            strengths=["Analysis completed despite review system issues"],
            weaknesses=[f"Unable to complete automated review: {error_msg}"],
            specific_improvements=["Manual review recommended"],
            should_continue=loop_iteration < max_loops,
            reason=f"Fallback review due to system error in loop {loop_iteration}/{max_loops}"
        )