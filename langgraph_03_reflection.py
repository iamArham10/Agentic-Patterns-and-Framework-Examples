"""
LANGGRAPH - REFLECTION PATTERN

The agent generates content, critiques it, and improves iteratively.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5)

# STEP 1: Define State

class ReflectionState(TypedDict):
    task: str                # What to write
    draft: str                # Current draft
    critique: str            # Feedback on draft
    iteration: int          # Current iteration
    max_iterations: int      # When to stop

# STEP 2: Define Nodes

def generate(state: ReflectionState) -> ReflectionState:
    """Node 1: Generate or improve content."""
    print(f"\n[NODE: generate] Iteration {state['iteration']}")
    
    if state["iteration"] == 0:
        prompt = f"Write a response for: {state['task']}\nBe concise (2-3 sentences)."
    else:
        prompt = f"""Improve this draft based on the feedback.

Task: {state['task']}

Current draft:
{state['draft']}

Feedback to address:
{state['critique']}

Write an improved version (2-3 sentences):"""
    
    response = llm.invoke(prompt)
    draft = response.content.strip()
    
    print(f"  Draft: {draft[:100]}...")
    
    return {**state, "draft": draft}

def critique(state: ReflectionState) -> ReflectionState:
    """Node 2: Critique the current draft."""
    print(f"\n[NODE: critique]")
    
    prompt = f"""Critique this draft. Find 1-2 specific issues to improve.

Task: {state['task']}

Draft:
{state['draft']}

Provide brief, actionable feedback:"""
    
    response = llm.invoke(prompt)
    critique = response.content.strip()
    
    print(f"  Critique: {critique[:100]}...")
    
    return {**state, "critique": critique, "iteration": state["iteration"] + 1}

# STEP 3: Define Routing

def should_continue(state: ReflectionState) -> str:
    """Decide: improve more or finish?"""
    if state["iteration"] < state["max_iterations"]:
        return "improve"    # Go back to generate
    return "finish"         # Done iterating

# STEP 4: Build the Graph

def build_reflection_agent():
    """Build the reflection agent."""
    graph = StateGraph(ReflectionState)
    
    # Add nodes
    graph.add_node("generate", generate)
    graph.add_node("critique", critique)
    
    # Add edges
    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", should_continue, {
        "improve": "generate",   # Loop back to improve
        "finish": END            # Done
    })
    
    return graph.compile()


if __name__ == "__main__":
    print("=" * 50)
    print("LANGGRAPH - REFLECTION PATTERN")
    print("=" * 50)
    print("""
Flow: Generate -> Critique -> Improve -> Critique -> ... -> Done

The agent improves its output through self-reflection.
    """)
    
    agent = build_reflection_agent()
    
    result = agent.invoke({
        "task": "Explain what an API is to a beginner",
        "draft": "",
        "critique": "",
        "iteration": 0,
        "max_iterations": 2  # Generate, then improve twice
    })
    
    print("\n" + "=" * 50)
    print("FINAL RESULT (after 2 improvement cycles):")
    print("=" * 50)
    print(result["draft"])
