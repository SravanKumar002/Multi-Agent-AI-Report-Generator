import os
from dotenv import load_dotenv
from typing import Literal, Dict, Any
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

# ================================
# Environment
# ================================

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Create Groq LLM
llm = init_chat_model("groq:llama-3.1-8b-instant")

# ================================
# State Definition
# ================================

from langgraph.graph import MessagesState

class AgentState(MessagesState):
    next_agent: str = ""
    research_data: str = ""
    market_data: str = ""
    merged_research: str = ""
    technical_text: str = ""
    summary_text: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""

# ================================
# CEO (Supervisor)
# ================================

def ceo_agent(state: AgentState) -> Dict[str, Any]:
    """CEO decides which team to activate."""

    has_research = bool(state.get("merged_research", ""))
    has_tech = bool(state.get("technical_text", ""))
    has_summary = bool(state.get("summary_text", ""))

    if not has_research:
        next_agent = "research_team_leader"
        ceo_msg = "🧑‍💼 CEO: Let's start research. Handing off to Research Team Leader."
    elif not (has_tech and has_summary):
        next_agent = "writing_team_leader"
        ceo_msg = "🧑‍💼 CEO: Research complete. Handing off to Writing Team Leader."
    else:
        next_agent = "end"
        ceo_msg = "🧑‍💼 CEO: All tasks complete. Great job team!"

    return {
        "messages": [AIMessage(content=ceo_msg)],
        "next_agent": next_agent,
    }

# ================================
# Research Team Leader
# ================================

def research_team_leader(state: AgentState) -> Dict[str, Any]:
    """Splits research between data and market researchers."""

    task = state.get("current_task", "No Task Provided")

    leader_msg = f"""📋 Research Team Leader:
Splitting research task:
- Data Researcher → factual data
- Market Researcher → market trends
Task: {task}"""

    return {
        "messages": [AIMessage(content=leader_msg)],
        "next_agent": "data_researcher"
    }

# ================================
# Data Researcher
# ================================

def data_researcher(state: AgentState) -> Dict[str, Any]:
    task = state.get("current_task", "")

    prompt = f"""As a Data Researcher, find factual data, statistics, and scientific info about:

{task}

Keep it concise but thorough."""

    result = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=f"🔎 Data Researcher:\n{result.content[:400]}...")],
        "research_data": result.content,
        "next_agent": "market_researcher",
    }

# ================================
# Market Researcher
# ================================

def market_researcher(state: AgentState) -> Dict[str, Any]:
    task = state.get("current_task", "")

    prompt = f"""As a Market Researcher, find trends, business insights, and market data about:

{task}

Keep it concise but thorough."""

    result = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=f"📈 Market Researcher:\n{result.content[:400]}...")],
        "market_data": result.content,
        "next_agent": "merge_research",
    }

# ================================
# Merge Research
# ================================

def merge_research(state: AgentState) -> Dict[str, Any]:
    """Merges Data + Market research."""

    merged = f"**DATA RESEARCH:**\n{state.get('research_data','')}\n\n**MARKET RESEARCH:**\n{state.get('market_data','')}"
    return {
        "messages": [AIMessage(content="✅ Research Team Leader: Merged research data.")],
        "merged_research": merged,
        "next_agent": "ceo_agent",
    }

# ================================
# Writing Team Leader
# ================================

def writing_team_leader(state: AgentState) -> Dict[str, Any]:
    """Splits writing into technical + summary writers."""

    msg = "📝 Writing Team Leader: Assigning tasks to Technical Writer and Summary Writer."
    return {
        "messages": [AIMessage(content=msg)],
        "next_agent": "technical_writer",
    }

# ================================
# Technical Writer
# ================================

def technical_writer(state: AgentState) -> Dict[str, Any]:
    research = state.get("merged_research", "")
    task = state.get("current_task", "")

    prompt = f"""As a Technical Writer, write a technical explanation about:

Task: {task}

Based on research:
{research[:2000]}
"""

    result = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=f"🧑‍💻 Technical Writer:\n{result.content[:400]}...")],
        "technical_text": result.content,
        "next_agent": "summary_writer",
    }

# ================================
# Summary Writer
# ================================

def summary_writer(state: AgentState) -> Dict[str, Any]:
    research = state.get("merged_research", "")
    task = state.get("current_task", "")

    prompt = f"""As a Summary Writer, create an executive summary for:

Task: {task}

Based on:
{research[:2000]}
"""

    result = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=f"📝 Summary Writer:\n{result.content[:400]}...")],
        "summary_text": result.content,
        "next_agent": "compile_report",
    }

# ================================
# Compile Report
# ================================

def compile_report(state: AgentState) -> Dict[str, Any]:
    report = f"""
📄 FINAL REPORT
=========================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Topic: {state.get('current_task','')}
=========================================================

**Technical Details:**
{state.get('technical_text','')}

**Executive Summary:**
{state.get('summary_text','')}

=========================================================
Compiled by Multi-Agent AI System powered by Groq
"""

    return {
        "messages": [AIMessage(content="✅ Writing Team Leader: Final report compiled.")],
        "final_report": report,
        "task_complete": True,
        "next_agent": "ceo_agent"
    }

# ================================
# Routing
# ================================

def router(state: AgentState) -> Literal[
    "ceo_agent",
    "research_team_leader",
    "data_researcher",
    "market_researcher",
    "merge_research",
    "writing_team_leader",
    "technical_writer",
    "summary_writer",
    "compile_report",
    "__end__"
]:
    if state.get("task_complete", False):
        return END

    return state.get("next_agent", "ceo_agent")

# ================================
# Build Graph
# ================================

workflow = StateGraph(AgentState)

workflow.add_node("ceo_agent", ceo_agent)
workflow.add_node("research_team_leader", research_team_leader)
workflow.add_node("data_researcher", data_researcher)
workflow.add_node("market_researcher", market_researcher)
workflow.add_node("merge_research", merge_research)
workflow.add_node("writing_team_leader", writing_team_leader)
workflow.add_node("technical_writer", technical_writer)
workflow.add_node("summary_writer", summary_writer)
workflow.add_node("compile_report", compile_report)

workflow.set_entry_point("ceo_agent")

for node in [
    "ceo_agent",
    "research_team_leader",
    "data_researcher",
    "market_researcher",
    "merge_research",
    "writing_team_leader",
    "technical_writer",
    "summary_writer",
    "compile_report",
]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "ceo_agent": "ceo_agent",
            "research_team_leader": "research_team_leader",
            "data_researcher": "data_researcher",
            "market_researcher": "market_researcher",
            "merge_research": "merge_research",
            "writing_team_leader": "writing_team_leader",
            "technical_writer": "technical_writer",
            "summary_writer": "summary_writer",
            "compile_report": "compile_report",
            END: END,
        },
    )

graph = workflow.compile()

# ================================
# Run Example
# ================================

if __name__ == "__main__":
    user_question = "What are the benefits and risks of AI in healthcare?"

    result = graph.invoke({
        "messages": [HumanMessage(content=user_question)],
        "current_task": user_question,
    })

    print(result.get("final_report"))   