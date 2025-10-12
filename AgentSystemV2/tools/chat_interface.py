"""
Interactive Chat Interface for post-analysis refinements
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from agents.domain_agent import DomainAgent
from agents.chat_agent import ChatAgent, create_report_chat_context
from config.schemas import ReviewResult, ChatRefinementRequest, ResearchReportOutput
from config.settings import (
    CHAT_MAX_LOOPS, MAX_CHAT_EXCHANGES, MAX_CHAT_REFINEMENTS,
    CHAT_HISTORY_WINDOW, MAX_CHAT_COST_PER_SESSION
)
from config.llm_cost_tracker import AgentCost, TokenUsage


class ChatInterface:
    """
    Interactive chat interface for post-analysis refinements.
    Allows users to ask follow-up questions and request analysis refinements.
    """
    
    def __init__(self, enable_debug: bool = False):
        """
        Initialize chat interface.
        
        Args:
            enable_debug: Enable debug logging
        """
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        if enable_debug:
            self.logger.setLevel(logging.DEBUG)
        
        # Chat session state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.chat_history: List[Dict[str, Any]] = []
        
        self.logger.info("ChatInterface initialized")
    
    async def start_chat_session(self,
                                session_id: str,
                                domain_agent: DomainAgent,
                                analysis_output: ResearchReportOutput,
                                review_history: List[ReviewResult]) -> Dict[str, Any]:
        """
        Start a new chat session for analysis refinement.
        
        Args:
            session_id: Unique session identifier
            domain_agent: Domain agent that produced the analysis
            analysis_output: Original analysis output
            review_history: History of review cycles
            
        Returns:
            Session initialization result
        """
        self.logger.info(f"Starting chat session: {session_id}")

        # Create compressed report context for ChatAgent with ticker mapping
        # Handle errors gracefully to allow ChatAgent to still function with minimal context
        try:
            report_context, allowed_tickers, ticker_to_name = create_report_chat_context(analysis_output)
            self.logger.debug(f"Created report context: {len(report_context)} characters")
            self.logger.debug(f"Allowed tickers: {allowed_tickers}")
            self.logger.debug(f"Ticker mapping: {ticker_to_name}")
        except Exception as e:
            self.logger.warning(f"Error creating report context: {e}. Using minimal fallback context.")
            # Provide minimal context to allow ChatAgent to still use RAG tools
            report_context = f"""
Company: {analysis_output.company}
Domain: {analysis_output.domain}
Confidence: {getattr(analysis_output, 'confidence_level', 'unknown')}

Note: Full report context unavailable due to processing error.
You can still use RAG tools to answer questions about {analysis_output.company}.
"""
            allowed_tickers = [analysis_output.company]
            ticker_to_name = {analysis_output.company: analysis_output.company}

        # Initialize ChatAgent for fast Q&A with company restrictions
        chat_agent = ChatAgent(
            domain=domain_agent.domain,
            company=analysis_output.company,
            report_context=report_context,
            allowed_tickers=allowed_tickers,
            ticker_to_name=ticker_to_name,
            enable_debug=self.enable_debug
        )

        # Create session state
        session_state = {
            "session_id": session_id,
            "domain": domain_agent.domain,
            "company": analysis_output.company,
            "domain_agent": domain_agent,
            "chat_agent": chat_agent,  # NEW: ChatAgent for Q&A
            "original_analysis": analysis_output,
            "review_history": review_history,
            "report_context": report_context,  # NEW: Compressed context
            "chat_history": [],
            "refinement_count": 0,
            "exchange_count": 0,  # NEW: Track total exchanges
            "total_cost": 0.0,  # NEW: Track session cost
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "status": "active"
        }

        self.active_sessions[session_id] = session_state
        
        # Create welcome message
        welcome_msg = self._create_welcome_message(analysis_output)
        
        self.logger.info(f"Chat session started for {analysis_output.company} - {domain_agent.domain}")
        
        return {
            "session_id": session_id,
            "status": "started",
            "welcome_message": welcome_msg,
            "available_commands": self._get_available_commands(),
            "analysis_summary": self._get_analysis_summary(analysis_output)
        }
    
    async def process_chat_message(self,
                                  session_id: str,
                                  user_message: str,
                                  refinement_type: str = "clarification") -> Dict[str, Any]:
        """
        Process user chat message and provide response using smart routing.

        Routes to either:
        - ChatAgent (fast, cheap) for simple Q&A
        - Full refinement (slow, expensive) for complex analysis requests

        Args:
            session_id: Chat session ID
            user_message: User's message/question
            refinement_type: Type of refinement requested

        Returns:
            Chat response with analysis or refinement
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found", "session_id": session_id}

        session = self.active_sessions[session_id]
        session["last_activity"] = datetime.now().isoformat()
        session["exchange_count"] += 1

        self.logger.info(f"Processing chat message in session {session_id} (exchange #{session['exchange_count']})")

        # Check session limits
        if session["exchange_count"] > MAX_CHAT_EXCHANGES:
            return {
                "type": "limit_reached",
                "message": f"Maximum exchanges ({MAX_CHAT_EXCHANGES}) reached for this session. Please start a new session.",
                "suggestion": "Start a new chat session to continue"
            }

        if session["total_cost"] >= MAX_CHAT_COST_PER_SESSION:
            return {
                "type": "cost_limit_reached",
                "message": f"Cost limit (${MAX_CHAT_COST_PER_SESSION:.2f}) reached for this session.",
                "suggestion": "Start a new session to continue chatting"
            }

        # Add user message to history
        user_entry = {
            "type": "user",
            "message": user_message,
            "refinement_type": refinement_type,
            "timestamp": datetime.now().isoformat()
        }
        session["chat_history"].append(user_entry)

        # Trim chat history if needed (sliding window)
        if len(session["chat_history"]) > CHAT_HISTORY_WINDOW * 2:  # user + assistant pairs
            # Keep only recent exchanges
            session["chat_history"] = session["chat_history"][-CHAT_HISTORY_WINDOW * 2:]
            self.logger.debug(f"Trimmed chat history to {len(session['chat_history'])} entries")

        try:
            # SMART ROUTING: Decide between fast Q&A vs expensive refinement
            if self._requires_refinement(user_message):
                # Route to expensive refinement path
                self.logger.info("Routing to full refinement (complex request)")
                response = await self._handle_refinement_request(session, user_message, refinement_type)
            else:
                # Route to fast ChatAgent path
                self.logger.info("Routing to ChatAgent (simple Q&A)")
                response = await self._handle_chat_question(session, user_message)

            # Add assistant response to history
            assistant_entry = {
                "type": "assistant",
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            session["chat_history"].append(assistant_entry)

            return response

        except Exception as e:
            self.logger.error(f"Chat processing failed: {e}")
            error_response = {
                "type": "error",
                "message": f"Sorry, I encountered an error: {str(e)}",
                "suggestions": ["Try rephrasing your question", "Ask for a specific metric or trend analysis"],
                "cost": 0.0
            }

            session["chat_history"].append({
                "type": "assistant",
                "response": error_response,
                "timestamp": datetime.now().isoformat()
            })

            return error_response
    
    def _requires_refinement(self, message: str) -> bool:
        """
        Determine if message requires expensive full refinement.

        Returns True for:
        - "Regenerate", "redo", "reanalyze" type requests
        - Complex multi-part questions
        - Requests to change persona or focus
        - Explicit refinement requests

        Returns False for:
        - Simple questions about existing report
        - Clarifications
        - Data lookups
        - Explanations
        """
        message_lower = message.lower()

        # Refinement indicators (expensive path)
        refinement_patterns = [
            "regenerate", "redo", "reanalyze", "rewrite",
            "change the focus", "different perspective", "different persona",
            "more comprehensive", "deeper analysis",
            "add more", "include additional",
            "refine the", "improve the",
            "reconsider", "reassess"
        ]

        if any(pattern in message_lower for pattern in refinement_patterns):
            return True

        # Simple Q&A indicators (cheap path)
        simple_patterns = [
            "what is", "what was", "what were",
            "how much", "how many",
            "when did", "when was",
            "explain", "clarify", "define",
            "tell me about", "show me",
            "what does", "why is", "why was",
            "can you explain", "could you explain",
            "help me understand"
        ]

        if any(pattern in message_lower for pattern in simple_patterns):
            return False

        # Default: treat as simple Q&A unless clearly a refinement request
        # This keeps costs low by default
        return False
    
    async def _handle_chat_question(self,
                                    session: Dict[str, Any],
                                    user_message: str) -> Dict[str, Any]:
        """
        Handle chat question using ChatAgent with RAG tools.

        Fast, cost-effective path for Q&A.
        """
        chat_agent = session["chat_agent"]
        chat_history = session["chat_history"]

        try:
            # Use ChatAgent to respond (with RAG tool access)
            response_text, tool_calls_made, token_usage = await chat_agent.respond_to_question(
                user_question=user_message,
                chat_history=chat_history
            )

            # Track cost
            cost = token_usage.calculate_cost()
            session["total_cost"] += cost

            self.logger.info(f"ChatAgent response cost: ${cost:.6f} ({token_usage.total_tokens} tokens)")

            # Extract tool names for metadata
            tools_used = [tc.get('function', 'unknown') for tc in tool_calls_made] if tool_calls_made else []

            return {
                "type": "chat_response",
                "message": response_text,
                "source": "chat_agent",
                "tokens_used": token_usage.total_tokens,
                "tools_used": tools_used,
                "cost": cost,
                "confidence": "high"
            }

        except Exception as e:
            self.logger.error(f"ChatAgent failed: {e}. Attempting fallback with RAG tools.")

            # Try to answer using RAG tools directly as fallback
            try:
                fallback_response = await self._answer_with_rag_tools(session, user_message)
                if fallback_response:
                    return fallback_response
            except Exception as rag_error:
                self.logger.error(f"RAG tools fallback also failed: {rag_error}")

            # Final fallback to simple extraction
            analysis = session["original_analysis"]
            response_text = self._extract_relevant_info(analysis, user_message)

            return {
                "type": "fallback_answer",
                "message": response_text,
                "source": "existing_analysis",
                "cost": 0.0,
                "note": "ChatAgent and RAG tools unavailable, using report extraction"
            }
    
    async def _answer_with_rag_tools(self,
                                     session: Dict[str, Any],
                                     user_message: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to answer question using RAG tools directly (fallback when ChatAgent fails).

        This is a simple fallback that tries to use RAG tools to answer the question
        even when ChatAgent is unavailable due to errors.
        """
        company = session["company"]
        domain = session["domain"]
        question_lower = user_message.lower()

        self.logger.info(f"Attempting to answer with RAG tools fallback for: {user_message[:50]}")

        # Import RAG tools
        try:
            from tools.rag_tools import (
                search_annual_reports,
                search_transcripts,
                search_news,
                get_financial_metrics,
                get_share_price_data
            )
        except ImportError as e:
            self.logger.error(f"Could not import RAG tools: {e}")
            return None

        tool_results = []
        tool_names_used = []

        try:
            # Determine which tool to use based on question keywords
            if any(word in question_lower for word in ["revenue", "income", "assets", "liabilities", "cash flow", "metric"]):
                # Try financial metrics
                metrics = ["revenue", "net_income", "total_assets", "operating_cash_flow"]
                result = get_financial_metrics(company, metrics, period="latest")
                if result.get("success"):
                    tool_results.append(f"Financial Metrics: {result['result']}")
                    tool_names_used.append("get_financial_metrics")

            if any(word in question_lower for word in ["filing", "10-k", "10-q", "annual report", "sec", "statement"]):
                # Try annual reports
                result = search_annual_reports(company, user_message, k=3, time_period="latest")
                if result.get("success"):
                    tool_results.append(f"Annual Reports: {result['result']}")
                    tool_names_used.append("search_annual_reports")

            if any(word in question_lower for word in ["earnings", "call", "transcript", "management", "cfo", "ceo", "guidance"]):
                # Try transcripts
                result = search_transcripts(company, user_message, quarters_back=2, k=3)
                if result.get("success"):
                    tool_results.append(f"Transcripts: {result['result']}")
                    tool_names_used.append("search_transcripts")

            if any(word in question_lower for word in ["news", "article", "announcement", "recent", "latest news"]):
                # Try news
                result = search_news(company, user_message, days_back=30)
                if result.get("success"):
                    tool_results.append(f"News: {result['result']}")
                    tool_names_used.append("search_news")

            if any(word in question_lower for word in ["stock", "price", "share", "market", "trading"]):
                # Try share price
                result = get_share_price_data(company, days_back=30)
                if result.get("success"):
                    tool_results.append(f"Share Price Data: {result['result']}")
                    tool_names_used.append("get_share_price_data")

            # If no specific tool matched, try a general search
            if not tool_results:
                # Try annual reports as default
                result = search_annual_reports(company, user_message, k=3, time_period="latest")
                if result.get("success"):
                    tool_results.append(f"Annual Reports: {result['result']}")
                    tool_names_used.append("search_annual_reports")

            # Format response
            if tool_results:
                response_text = f"Based on RAG tool search for {company} regarding '{user_message}':\n\n"
                response_text += "\n\n".join(tool_results)
                response_text += f"\n\n(Sources: {', '.join(tool_names_used)})"

                return {
                    "type": "rag_fallback_response",
                    "message": response_text,
                    "source": "rag_tools_direct",
                    "tools_used": tool_names_used,
                    "cost": 0.0,  # RAG tools don't incur LLM costs
                    "note": "Answered using RAG tools directly (ChatAgent unavailable)"
                }
            else:
                self.logger.warning("No RAG tool results obtained")
                return None

        except Exception as e:
            self.logger.error(f"Error in _answer_with_rag_tools: {e}")
            return None

    async def _handle_refinement_request(self,
                                        session: Dict[str, Any],
                                        user_message: str,
                                        refinement_type: str) -> Dict[str, Any]:
        """
        Handle complex refinement requests that require new analysis.

        Expensive path - runs full analyst-reviewer loop.
        """
        session["refinement_count"] += 1

        # Check refinement limit
        if session["refinement_count"] > MAX_CHAT_REFINEMENTS:
            return {
                "type": "limit_reached",
                "message": f"Maximum refinements ({MAX_CHAT_REFINEMENTS}) reached for this session. Please start a new analysis if you need extensive changes.",
                "suggestion": "Consider starting a new domain analysis with updated focus",
                "cost": 0.0
            }

        self.logger.info(f"Performing refinement {session['refinement_count']} in session {session['session_id']}")

        # Create refinement request
        refinement_request = ChatRefinementRequest(
            original_analysis=session["original_analysis"],
            user_question=user_message,
            refinement_type=refinement_type,
            max_loops=CHAT_MAX_LOOPS
        )

        # Perform refinement analysis (expensive!)
        domain_agent = session["domain_agent"]
        refined_analysis, review_history, run_cost = await domain_agent.chat_refinement(refinement_request)

        # Track cost
        refinement_cost = run_cost.total_cost
        session["total_cost"] += refinement_cost

        self.logger.info(f"Refinement cost: ${refinement_cost:.4f}")

        # Update session with refined analysis
        session["latest_analysis"] = refined_analysis
        session["latest_review_history"] = review_history

        # Update ChatAgent with new context (including updated ticker mapping)
        new_report_context, new_allowed_tickers, new_ticker_to_name = create_report_chat_context(refined_analysis)
        session["chat_agent"].update_report_context(new_report_context)
        session["chat_agent"].allowed_tickers = new_allowed_tickers
        session["chat_agent"].ticker_to_name = new_ticker_to_name
        session["report_context"] = new_report_context

        # Extract RAG tool sources from the refined analysis
        sources_list = self._extract_sources_from_analysis(refined_analysis)
        source_summary = self._format_sources_summary(sources_list)

        return {
            "type": "refinement",
            "message": refined_analysis.executive_summary,
            "source": source_summary,
            "sources_list": sources_list,  # Detailed list of sources
            "analysis_bullets_count": len(refined_analysis.analysis_bullets) if hasattr(refined_analysis, 'analysis_bullets') else 0,
            "confidence": refined_analysis.confidence_level,
            "refinement_number": session["refinement_count"],
            "review_cycles": len(review_history),
            "quality_score": review_history[-1].quality_score if review_history else 0.0,
            "cost": refinement_cost,
            "total_session_cost": session["total_cost"]
        }
    
    def _extract_sources_from_analysis(self, analysis: ResearchReportOutput) -> List[Dict[str, Any]]:
        """
        Extract all RAG tool sources from analysis bullets.

        Returns a list of source dictionaries with tool name, period, and other metadata.
        """
        sources = []

        if not hasattr(analysis, 'analysis_bullets') or not analysis.analysis_bullets:
            return sources

        # Collect all qualitative sources from all bullets
        for bullet in analysis.analysis_bullets:
            if hasattr(bullet, 'qualitative_sources') and bullet.qualitative_sources:
                for source in bullet.qualitative_sources:
                    # Handle both DetailedSource objects and dicts
                    if hasattr(source, 'to_dict'):
                        source_dict = source.to_dict()
                    elif isinstance(source, dict):
                        source_dict = source
                    else:
                        continue

                    # Extract key information
                    tool_name = source_dict.get('tool_name', 'unknown')
                    period = source_dict.get('period', '')
                    speaker = source_dict.get('speaker_name', '')
                    report_type = source_dict.get('report_type', '')

                    # Create source entry
                    source_entry = {
                        'tool_name': tool_name,
                        'period': period,
                        'speaker_name': speaker,
                        'report_type': report_type,
                        'relevance_score': source_dict.get('relevance_score', 0.0)
                    }

                    sources.append(source_entry)

        return sources

    def _format_sources_summary(self, sources: List[Dict[str, Any]]) -> str:
        """
        Format sources list into a concise summary string.

        Example output: "Transcripts (Q3-2025), Annual Reports (Q1-2024, Q2-2024), Financial RAG"
        """
        if not sources:
            return "Analysis only (no external sources)"

        # Group sources by tool
        tool_groups = {}
        for source in sources:
            tool = source['tool_name']
            period = source.get('period', '')
            report_type = source.get('report_type', '')
            speaker = source.get('speaker_name', '')

            if tool not in tool_groups:
                tool_groups[tool] = []

            # Format source detail
            if period:
                detail = period
                if report_type:
                    detail = f"{report_type} {period}"
                elif speaker and speaker != 'Unknown':
                    detail = f"{period} ({speaker})"
                tool_groups[tool].append(detail)

        # Create human-readable summary
        summary_parts = []

        # Map tool names to friendly names
        tool_name_map = {
            'search_transcripts': 'Transcripts',
            'search_annual_reports': 'Annual Reports',
            'search_news': 'News',
            'get_financial_metrics': 'Financial RAG',
            'compare_companies': 'Company Comparison',
            'get_share_price_data': 'Share Price Data'
        }

        for tool, details in tool_groups.items():
            friendly_name = tool_name_map.get(tool, tool.replace('_', ' ').title())

            # Get unique details
            unique_details = list(dict.fromkeys(details))  # Preserve order, remove duplicates

            if unique_details and any(d for d in unique_details if d):  # Has meaningful details
                # Limit to 3 most recent
                detail_str = ', '.join(unique_details[:3])
                summary_parts.append(f"{friendly_name} ({detail_str})")
            else:
                summary_parts.append(friendly_name)

        return ', '.join(summary_parts)

    def _extract_relevant_info(self, analysis: ResearchReportOutput, question: str) -> str:
        """Extract relevant information from existing analysis."""

        question_lower = question.lower()

        # Simple keyword matching to find relevant information
        if "summary" in question_lower or "overview" in question_lower:
            return analysis.executive_summary

        elif "findings" in question_lower or "key" in question_lower:
            # Extract bullet text from analysis_bullets
            if hasattr(analysis, 'analysis_bullets') and analysis.analysis_bullets:
                bullet_texts = [bullet.bullet_text for bullet in analysis.analysis_bullets]
                return "Key findings: " + "; ".join(bullet_texts)
            else:
                return "Key findings not available in this analysis."

        elif "confidence" in question_lower:
            return f"Analysis confidence level: {analysis.confidence_level}"

        elif "risk" in question_lower:
            return "Risk factors: " + "; ".join(analysis.risk_factors) if analysis.risk_factors else "No specific risk factors identified."

        elif "recommend" in question_lower:
            return "Recommendations: " + "; ".join(analysis.recommendations) if analysis.recommendations else "No specific recommendations provided."

        else:
            # Return a general summary
            return f"Based on the {analysis.domain} analysis of {analysis.company}: {analysis.executive_summary[:200]}..."
    
    def _create_welcome_message(self, analysis: ResearchReportOutput) -> str:
        """Create welcome message for chat session."""
        
        return f"""
Welcome to the interactive analysis chat for {analysis.company}!

I've completed a {analysis.domain} analysis with {analysis.confidence_level} confidence.
You can now ask follow-up questions or request refinements.

Examples of what you can ask:
- "Explain the current ratio trend in more detail"
- "What are the main liquidity risks?"
- "How does this compare to industry averages?"
- "Provide more analysis on working capital efficiency"

Type your question or request below.
"""
    
    def _get_available_commands(self) -> List[str]:
        """Get list of available chat commands."""
        
        return [
            "Ask specific questions about metrics or ratios",
            "Request deeper analysis on particular areas",
            "Ask for clarification on findings",
            "Request alternative perspective or analysis",
            "Ask about peer comparisons",
            "Request risk assessment details"
        ]
    
    def _get_analysis_summary(self, analysis: ResearchReportOutput) -> Dict[str, Any]:
        """Get concise analysis summary for chat context."""

        return {
            "domain": analysis.domain,
            "company": analysis.company,
            "confidence": analysis.confidence_level,
            "analysis_bullets_count": len(analysis.analysis_bullets) if hasattr(analysis, 'analysis_bullets') else 0,
            "risk_factors_count": len(analysis.risk_factors),
            "recommendations_count": len(analysis.recommendations),
            "analysis_timestamp": analysis.analysis_timestamp
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of chat session with cost tracking."""

        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]

        return {
            "session_id": session_id,
            "status": session["status"],
            "company": session["company"],
            "domain": session["domain"],
            "exchange_count": session.get("exchange_count", 0),
            "chat_exchanges": len(session["chat_history"]) // 2,  # User-assistant pairs
            "refinements_used": session["refinement_count"],
            "total_cost": session.get("total_cost", 0.0),
            "cost_limit": MAX_CHAT_COST_PER_SESSION,
            "exchange_limit": MAX_CHAT_EXCHANGES,
            "last_activity": session["last_activity"],
            "created_at": session["created_at"],
            "has_chat_agent": "chat_agent" in session
        }
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End chat session and cleanup."""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session = self.active_sessions[session_id]
        session["status"] = "ended"
        session["ended_at"] = datetime.now().isoformat()
        
        # Archive session
        self.chat_history.append(session)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        self.logger.info(f"Chat session ended: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "ended",
            "total_exchanges": len(session["chat_history"]) // 2,
            "refinements_performed": session["refinement_count"]
        }
    
    def get_all_active_sessions(self) -> List[Dict[str, Any]]:
        """Get summary of all active chat sessions."""
        
        summaries = []
        for session_id, session in self.active_sessions.items():
            summaries.append({
                "session_id": session_id,
                "company": session["company"],
                "domain": session["domain"],
                "exchanges": len(session["chat_history"]) // 2,
                "refinements": session["refinement_count"],
                "last_activity": session["last_activity"]
            })
        
        return summaries