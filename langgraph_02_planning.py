"""
LANGGRAPH - PLANNING PATTERN

The agent creates a plan first, then executes each step.
LangGraph manages the state and flow between planning and execution.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# STEP 1: Define State

class PlanState(TypedDict):
    task: str                # Original task
    plan: list              # List of steps
    current_step: int       # Which step we're on
    results: list           # Results from each step
    final_answer: str       # Final combined answer

# STEP 2: Define Nodes

def create_plan(state: PlanState) -> PlanState:
    """Node 1: Break task into steps."""
    print("\n[NODE: create_plan]")
    
    prompt = f"""Break this task into 3 simple steps.
Task: {state['task']}

Return ONLY numbered steps like:
1. First step
2. Second step  
3. Third step"""
    
    response = llm.invoke(prompt)
    
    # Parse steps
    steps = []
    for line in response.content.strip().split('\n'):
        if line.strip() and line.strip()[0].isdigit():
            step = line.split('.', 1)[-1].strip()
            steps.append(step)
    
    print(f"  Plan created: {len(steps)} steps")
    for i, s in enumerate(steps, 1):
        print(f"    {i}. {s}")
    
    return {**state, "plan": steps, "current_step": 0, "results": []}

def execute_step(state: PlanState) -> PlanState:
    """Node 2: Execute current step."""
    step_num = state["current_step"]
    step = state["plan"][step_num]
    
    print(f"\n[NODE: execute_step] Step {step_num + 1}")
    
    context = "\n".join(state["results"]) if state["results"] else "None"
    prompt = f"""Execute this step: {step}
Previous results: {context}
Be brief (1-2 sentences):"""
    
    response = llm.invoke(prompt)
    result = response.content.strip()
    
    print(f"  Step: {step}")
    print(f"  Result: {result[:100]}...")
    
    new_results = state["results"] + [f"Step {step_num + 1}: {result}"]
    
    return {**state, "current_step": step_num + 1, "results": new_results}

def summarize(state: PlanState) -> PlanState:
    """Node 3: Combine all results into final answer."""
    print("\n[NODE: summarize]")
    
    all_results = "\n".join(state["results"])
    prompt = f"""Task: {state['task']}

Step results:
{all_results}

Provide a brief final summary:"""
    
    response = llm.invoke(prompt)
    print(f"  Final: {response.content[:150]}...")
    
    return {**state, "final_answer": response.content}

# STEP 3: Define Routing

def should_continue(state: PlanState) -> str:
    """Check if more steps to execute."""
    if state["current_step"] < len(state["plan"]):
        return "continue"      # More steps to do
    return "summarize"         # All steps done

# STEP 4: Build the Graph

def build_planner():
    """Build the planning agent."""
    graph = StateGraph(PlanState)
    
    # Add nodes
    graph.add_node("create_plan", create_plan)
    graph.add_node("execute_step", execute_step)
    graph.add_node("summarize", summarize)
    
    # Add edges
    graph.set_entry_point("create_plan")
    graph.add_edge("create_plan", "execute_step")
    graph.add_conditional_edges("execute_step", should_continue, {
        "continue": "execute_step",   # Loop back for next step
        "summarize": "summarize"      # Done with steps
    })
    graph.add_edge("summarize", END)
    
    return graph.compile()

if __name__ == "__main__":
    print("=" * 50)
    print("LANGGRAPH - PLANNING PATTERN")
    print("=" * 50)
    
    planner = build_planner()
    
    result = planner.invoke({
        "task": "Explain how to make a sandwich",
        "plan": [],
        "current_step": 0,
        "results": [],
        "final_answer": ""
    })
    
    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print("=" * 50)
    print(result["final_answer"])
