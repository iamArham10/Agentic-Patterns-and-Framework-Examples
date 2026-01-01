"""
LANGGRAPH - TOOL USAGE PATTERN

LangGraph lets you build stateful agents as graphs.
- Nodes = steps (functions)
- Edges = flow between steps
- State = shared data between nodes
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ============================================================
# STEP 1: Define Tools (functions the agent can use)
# ============================================================

@tool
def calculator(expression: str) -> str:
    """Calculate a math expression. Example: calculator("2 + 2")"""
    try:
        return f"Result: {eval(expression)}"
    except:
        return "Error calculating"

@tool
def get_weather(city: str) -> str:
    """Get weather for a city. Example: get_weather("London")"""
    # Simulated - in real app, call weather API
    return f"Weather in {city}: Sunny, 22°C"

TOOLS = [calculator, get_weather]

# ============================================================
# STEP 2: Define State (shared data between nodes)
# ============================================================

class AgentState(TypedDict):
    messages: list          # Conversation history
    tool_result: str        # Result from tool execution

# ============================================================
# STEP 3: Define Nodes (steps in the graph)
# ============================================================

# Create LLM with tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
llm_with_tools = llm.bind_tools(TOOLS)

def call_llm(state: AgentState) -> AgentState:
    """Node 1: Call LLM - it may request a tool."""
    print("\n[NODE: call_llm]")
    response = llm_with_tools.invoke(state["messages"])
    print(f"  LLM says: {response.content[:100] if response.content else '(requesting tool)'}")
    return {"messages": state["messages"] + [response], "tool_result": state.get("tool_result", "")}

def execute_tool(state: AgentState) -> AgentState:
    """Node 2: Execute the tool LLM requested."""
    print("[NODE: execute_tool]")
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Find and execute the tool
        tool_map = {t.name: t for t in TOOLS}
        if tool_name in tool_map:
            result = tool_map[tool_name].invoke(tool_args)
            print(f"  Executed {tool_name}({tool_args}) = {result}")
            return {"messages": state["messages"], "tool_result": result}
    
    return state

def respond(state: AgentState) -> AgentState:
    """Node 3: Give final response with tool result."""
    print("[NODE: respond]")
    if state.get("tool_result"):
        final = llm.invoke(state["messages"] + [HumanMessage(content=f"Tool returned: {state['tool_result']}. Give a final answer.")])
        print(f"  Final: {final.content}")
    return state

# ============================================================
# STEP 4: Define Routing (which node to go next)
# ============================================================

def should_use_tool(state: AgentState) -> str:
    """Decide: did LLM request a tool?"""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "use_tool"      # Go to execute_tool node
    return "end"               # Go to END

# ============================================================
# STEP 5: Build the Graph
# ============================================================

def build_agent():
    """Build the LangGraph agent."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("respond", respond)
    
    # Add edges (flow)
    graph.set_entry_point("call_llm")
    graph.add_conditional_edges("call_llm", should_use_tool, {
        "use_tool": "execute_tool",
        "end": END
    })
    graph.add_edge("execute_tool", "respond")
    graph.add_edge("respond", END)
    
    return graph.compile()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("LANGGRAPH - TOOL USAGE PATTERN")
    print("=" * 50)
    
    agent = build_agent()
    
    print("\nQuery: What is 14 * 7?")
    result = agent.invoke({
        "messages": [HumanMessage(content="What is 14 * 7?")],
        "tool_result": ""
    })
    
    print("\n" + "-" * 50)
    
    print("\nQuery: What's the weather in Lahore?")
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Lahore?")],
        "tool_result": ""
    })
