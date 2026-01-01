# Code Explanation Guide

This document explains what's happening in each code file step-by-step.

---

## Table of Contents
1. [LangGraph: Tool Usage Pattern](#1-langgraph-tool-usage-pattern)
2. [LangGraph: Planning Pattern](#2-langgraph-planning-pattern)
3. [LangGraph: Reflection Pattern](#3-langgraph-reflection-pattern)
4. [CrewAI: Multi-Agent Pattern](#4-crewai-multi-agent-pattern)
5. [CrewAI: Memory Pattern](#5-crewai-memory-pattern)

---

## 1. LangGraph: Tool Usage Pattern

**File:** `langgraph_01_tools.py`

### What is LangGraph?

LangGraph lets you build AI agents as **graphs**:
- **Nodes** = Functions (steps in your workflow)
- **Edges** = Connections between nodes (the flow)
- **State** = Shared data that passes between nodes

Think of it like a flowchart that the code follows.

### Code Breakdown

#### Step 1: Define Tools

```python
@tool
def calculator(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"
```

**What's happening:**
- `@tool` decorator tells LangChain "this function can be used by the LLM"
- The docstring is important! LLM reads it to know WHEN to use the tool
- These are just regular Python functions

#### Step 2: Define State

```python
class AgentState(TypedDict):
    messages: list          # Conversation history
    tool_result: str        # Result from tool execution
```

**What's happening:**
- State is like a "shared clipboard" between all nodes
- Every node can read from and write to this state
- `messages` keeps track of the conversation
- `tool_result` stores what the tool returned

#### Step 3: Define Nodes (Functions)

```python
def call_llm(state: AgentState) -> AgentState:
    """Node 1: Ask LLM what to do"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response], ...}

def execute_tool(state: AgentState) -> AgentState:
    """Node 2: Run the tool LLM requested"""
    tool_call = state["messages"][-1].tool_calls[0]
    result = TOOLS[tool_call["name"]].invoke(tool_call["args"])
    return {..., "tool_result": result}

def respond(state: AgentState) -> AgentState:
    """Node 3: Give final answer"""
    # Uses tool_result to form final response
```

**What's happening:**
- Each node is a function that takes state and returns updated state
- `call_llm`: Asks the LLM, LLM might say "I need to use calculator"
- `execute_tool`: Actually runs the calculator function
- `respond`: Gives the final answer to the user

#### Step 4: Define Routing

```python
def should_use_tool(state: AgentState) -> str:
    """Decide which path to take"""
    if state["messages"][-1].tool_calls:
        return "use_tool"    # LLM wants a tool
    return "end"             # LLM answered directly
```

**What's happening:**
- This is the "decision point" in the graph
- Checks: Did the LLM request a tool?
- Returns a string that matches an edge name

#### Step 5: Build the Graph

```python
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
```

**What's happening:**
- We're building a flowchart in code
- `set_entry_point`: Where to start
- `add_conditional_edges`: IF-ELSE logic (if tool needed, go here; else, go there)
- `add_edge`: Simple A → B connection

### Visual Flow

```
                    ┌─────────────┐
                    │  call_llm   │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │ should_use_tool?        │
              ▼                         ▼
        "use_tool"                    "end"
              │                         │
              ▼                         │
        ┌───────────────┐               │
        │ execute_tool  │               │
        └───────┬───────┘               │
                │                       │
                ▼                       │
        ┌───────────────┐               │
        │   respond     │               │
        └───────┬───────┘               │
                │                       │
                └───────────┬───────────┘
                            ▼
                          [END]
```

---

## 2. LangGraph: Planning Pattern

**File:** `langgraph_02_planning.py`

### The Idea

Instead of trying to answer everything at once:
1. **Plan**: Break the task into steps
2. **Execute**: Do each step one by one
3. **Summarize**: Combine results

### Code Breakdown

#### State Definition

```python
class PlanState(TypedDict):
    task: str                # "Explain how to make a sandwich"
    plan: list              # ["Get ingredients", "Assemble", "Serve"]
    current_step: int       # 0, 1, 2... (which step we're on)
    results: list           # Results from completed steps
    final_answer: str       # Combined final answer
```

**What's happening:**
- More complex state than before
- `current_step` is like a counter - tracks progress
- `results` accumulates output from each step

#### Node 1: Create Plan

```python
def create_plan(state: PlanState) -> PlanState:
    prompt = f"Break this task into 3 steps: {state['task']}"
    response = llm.invoke(prompt)
    
    # Parse "1. First\n2. Second\n3. Third" into a list
    steps = [parse each line...]
    
    return {**state, "plan": steps, "current_step": 0}
```

**What's happening:**
- Asks LLM to create a plan
- Parses the numbered list into a Python list
- Sets `current_step` to 0 (start at beginning)

#### Node 2: Execute Step

```python
def execute_step(state: PlanState) -> PlanState:
    step = state["plan"][state["current_step"]]  # Get current step
    
    prompt = f"Execute: {step}\nPrevious results: {state['results']}"
    result = llm.invoke(prompt)
    
    return {
        **state, 
        "current_step": state["current_step"] + 1,  # Move to next
        "results": state["results"] + [result]       # Save result
    }
```

**What's happening:**
- Gets the current step from the plan
- Executes it (asks LLM)
- Increments `current_step` (moves forward)
- Adds result to `results` list

#### Routing: Should Continue?

```python
def should_continue(state: PlanState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "continue"    # More steps to do → loop back
    return "summarize"       # All done → go to summarize
```

**What's happening:**
- Simple check: Are there more steps?
- If yes: loop back to `execute_step`
- If no: go to `summarize`

#### The Loop

```python
graph.add_conditional_edges("execute_step", should_continue, {
    "continue": "execute_step",   # ← LOOPS BACK TO ITSELF!
    "summarize": "summarize"
})
```

**What's happening:**
- This creates a LOOP in the graph
- `execute_step` can go back to itself
- This is powerful - LangGraph handles loops cleanly

### Visual Flow

```
┌─────────────┐
│ create_plan │
└──────┬──────┘
       │
       ▼
┌──────────────┐ ◄─────────┐
│ execute_step │           │
└──────┬───────┘           │
       │                   │
       ▼                   │
  more steps? ─── yes ─────┘
       │
       no
       │
       ▼
┌───────────┐
│ summarize │
└─────┬─────┘
      │
      ▼
    [END]
```

---

## 3. LangGraph: Reflection Pattern

**File:** `langgraph_03_reflection.py`

### The Idea

The agent improves its output through self-critique:
1. **Generate**: Create initial response
2. **Critique**: Find issues and suggest improvements
3. **Improve**: Rewrite based on feedback
4. **Repeat**: Until max iterations reached

### Code Breakdown

#### State Definition

```python
class ReflectionState(TypedDict):
    task: str                # "Explain what an API is"
    draft: str               # Current version of the response
    critique: str            # Feedback on the draft
    iteration: int           # 0, 1, 2... (how many times we've improved)
    max_iterations: int      # When to stop (e.g., 2)
```

**What's happening:**
- `draft` holds the current version (gets better each iteration)
- `critique` stores the feedback
- `iteration` counts how many improvement cycles we've done

#### Node 1: Generate

```python
def generate(state: ReflectionState) -> ReflectionState:
    if state["iteration"] == 0:
        # First time: generate from scratch
        prompt = f"Write a response for: {state['task']}"
    else:
        # Later: improve based on critique
        prompt = f"""Improve this draft based on feedback.
        Draft: {state['draft']}
        Feedback: {state['critique']}
        Write an improved version:"""
    
    response = llm.invoke(prompt)
    return {**state, "draft": response.content}
```

**What's happening:**
- First iteration (0): Creates initial draft
- Later iterations: Uses critique to improve
- Same node, different behavior based on iteration count

#### Node 2: Critique

```python
def critique(state: ReflectionState) -> ReflectionState:
    prompt = f"""Critique this draft. Find 1-2 issues to improve.
    Task: {state['task']}
    Draft: {state['draft']}
    Provide brief, actionable feedback:"""
    
    response = llm.invoke(prompt)
    return {
        **state, 
        "critique": response.content,
        "iteration": state["iteration"] + 1  # Increment counter
    }
```

**What's happening:**
- Reviews the current draft
- Identifies specific issues
- Increments iteration counter

#### Routing: Should Continue?

```python
def should_continue(state: ReflectionState) -> str:
    if state["iteration"] < state["max_iterations"]:
        return "improve"    # Loop back to generate
    return "finish"         # Done
```

**What's happening:**
- Checks: Have we done enough iterations?
- If no: go back to `generate` node (improve)
- If yes: end the workflow

#### The Reflection Loop

```python
graph.set_entry_point("generate")
graph.add_edge("generate", "critique")
graph.add_conditional_edges("critique", should_continue, {
    "improve": "generate",   # ← LOOPS BACK!
    "finish": END
})
```

**What's happening:**
- Generate → Critique → (maybe) Generate → Critique → ... → END
- The loop continues until `max_iterations` is reached

### Visual Flow

```
┌──────────┐
│ generate │ ◄─────────┐
└────┬─────┘           │
     │                 │
     ▼                 │
┌──────────┐           │
│ critique │           │
└────┬─────┘           │
     │                 │
     ▼                 │
 more iterations? ─yes─┘
     │
     no
     │
     ▼
   [END]
```

### Example Run

```
Iteration 0:
  Generate: "An API is a way for programs to talk to each other."
  Critique: "Too vague. Add a concrete example."

Iteration 1:
  Generate: "An API is a way for programs to talk to each other. 
            For example, a weather app uses an API to get data from a weather service."
  Critique: "Good example. Could explain WHY APIs are useful."

Iteration 2:
  Generate: "An API is a way for programs to talk to each other.
            For example, a weather app uses an API to get data from a weather service.
            APIs save developers time by letting them use existing services instead of building everything from scratch."

Final output is much better than iteration 0!
```

---

## 4. CrewAI: Multi-Agent Pattern

**File:** `crewai_01_multiagent.py`

### What is CrewAI?

CrewAI makes it easy to create **teams of AI agents** that work together.

Key concepts:
- **Agent**: A persona with a role, goal, and backstory
- **Task**: Work assigned to an agent
- **Crew**: A team of agents working together

### Code Breakdown

#### Step 1: Configure LLM

```python
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)
```

**What's happening:**
- CrewAI has its own `LLM` class
- Format: `"provider/model-name"`
- Uses your Gemini API key from `.env`

#### Step 2: Define Agents

```python
researcher = Agent(
    role="Researcher",
    goal="Find key facts and information about topics",
    backstory="You are an expert researcher who finds accurate information.",
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Writer", 
    goal="Write clear, engaging content based on research",
    backstory="You are a skilled writer who creates easy-to-read content.",
    llm=llm,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Polish and improve written content",
    backstory="You are a meticulous editor who ensures quality.",
    llm=llm,
    verbose=True
)
```

**What's happening:**
- Each agent is like a persona
- `role`: Job title (used in prompts)
- `goal`: What they're trying to achieve
- `backstory`: Personality/expertise (helps LLM act the part)
- `verbose=True`: Print what the agent is doing

**Why backstory matters:**
The backstory shapes HOW the agent responds. A "meticulous editor" will be more critical than a "friendly helper".

#### Step 3: Define Tasks

```python
research_task = Task(
    description="Research the topic: {topic}. Find 3-4 key facts.",
    expected_output="A list of 3-4 key facts about the topic",
    agent=researcher  # ← Assigned to researcher
)

writing_task = Task(
    description="Write a short article about: {topic}",
    expected_output="A 2-3 paragraph article",
    agent=writer      # ← Assigned to writer
)

editing_task = Task(
    description="Edit and polish the article.",
    expected_output="A polished, publication-ready article",
    agent=editor      # ← Assigned to editor
)
```

**What's happening:**
- Each task has a description (what to do)
- `expected_output`: Helps the agent know what format to produce
- `agent`: WHO does this task

#### Step 4: Create and Run Crew

```python
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    verbose=True
)

result = crew.kickoff()  # ← Starts the work!
```

**What's happening:**
- Crew combines agents and tasks
- `kickoff()` runs everything
- Tasks execute in order: research → write → edit
- Output from task 1 is available to task 2, etc.

### Visual Flow

```
                    CREW
    ┌─────────────────────────────────┐
    │                                 │
    │  ┌──────────┐                   │
    │  │Researcher│ → research_task   │
    │  └────┬─────┘                   │
    │       │ (output)                │
    │       ▼                         │
    │  ┌──────────┐                   │
    │  │  Writer  │ → writing_task    │
    │  └────┬─────┘                   │
    │       │ (output)                │
    │       ▼                         │
    │  ┌──────────┐                   │
    │  │  Editor  │ → editing_task    │
    │  └────┬─────┘                   │
    │       │                         │
    └───────┼─────────────────────────┘
            │
            ▼
      Final Result
```

---

## 5. CrewAI: Memory Pattern

**File:** `crewai_02_memory.py`

### The Idea

Memory lets agents **remember** information across interactions:
- User preferences
- Past conversations
- Facts about entities

### Code Breakdown

#### Enable Memory on Agent

```python
assistant = Agent(
    role="Personal Assistant",
    goal="Help users and remember their preferences",
    backstory="You are a helpful assistant with excellent memory...",
    llm=llm,
    memory=True,  # ← THIS ENABLES MEMORY
    verbose=True
)
```

**What's happening:**
- `memory=True` tells the agent to remember things
- CrewAI handles the storage automatically

#### Enable Memory on Crew

```python
crew = Crew(
    agents=[assistant],
    tasks=[],
    memory=True,  # ← CREW-LEVEL MEMORY
    verbose=True
)
```

**What's happening:**
- Crew-level memory is shared across all agents
- Persists across multiple `kickoff()` calls

#### The Memory Demo

```python
# Interaction 1: Give information
task1 = Task(description="User says: 'My name is Alex and I love Python'")
crew.tasks = [task1]
crew.kickoff()  # Agent learns: name=Alex, likes=Python

# Interaction 2: Test recall
task2 = Task(description="User asks: 'What language do I like?'")
crew.tasks = [task2]
crew.kickoff()  # Agent should remember: Python!

# Interaction 3: Add more info
task3 = Task(description="User says: 'I also enjoy ML projects'")
crew.tasks = [task3]
crew.kickoff()  # Agent adds: interests=ML

# Interaction 4: Test combined memory
task4 = Task(description="User asks: 'Suggest a project for me'")
crew.tasks = [task4]
crew.kickoff()  # Agent uses ALL remembered info
```

**What's happening:**
- Each `kickoff()` is a separate interaction
- But memory persists between them
- Agent builds up knowledge about the user

### Types of Memory in CrewAI

| Type | Description | Example |
|------|-------------|---------|
| **Short-term** | Current conversation | "User just asked about Python" |
| **Long-term** | Persists across sessions | "User prefers dark mode" |
| **Entity** | Facts about specific things | "Alex likes Python and ML" |

### Visual Flow

```
Interaction 1          Interaction 2          Interaction 3
     │                      │                      │
     ▼                      ▼                      ▼
┌─────────┐           ┌─────────┐           ┌─────────┐
│  Agent  │           │  Agent  │           │  Agent  │
└────┬────┘           └────┬────┘           └────┬────┘
     │                      │                      │
     ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────┐
│                    MEMORY STORE                      │
│  - name: Alex                                        │
│  - likes: Python                                     │
│  - interests: ML                                     │
└─────────────────────────────────────────────────────┘
```

---

## Summary: When to Use What

| Pattern | Use When | Framework |
|---------|----------|-----------|
| **Tool Usage** | Agent needs to call APIs, calculate, search | LangGraph |
| **Planning** | Complex task needs step-by-step approach | LangGraph |
| **Multi-Agent** | Different expertise needed (researcher, writer, etc.) | CrewAI |
| **Memory** | Need to remember user preferences, context | CrewAI |
| **Reflection** | Need high-quality output through iteration | n8n (visual) |

---

## Key Takeaways

### LangGraph
- Think in **graphs**: nodes (functions) + edges (flow)
- **State** is the shared data between nodes
- **Conditional edges** = decision points
- Great for **loops** and **complex flows**

### CrewAI
- Think in **teams**: agents with roles
- **Agents** = personas with goals
- **Tasks** = work assigned to agents
- **Crew** = team that executes tasks
- Great for **collaboration** and **memory**
