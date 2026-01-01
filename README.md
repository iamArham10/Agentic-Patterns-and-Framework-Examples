# AI Agentic Design Patterns - Educational Examples

Examples of 5 agentic patterns using **LangGraph**, **CrewAI**, and **n8n**.

## Setup

### 1. Get Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** (top left)
4. Click **"Create API Key"** and select a project
5. Copy the key

**Free Tier Limits:** 15 requests/min, 1500 requests/day - plenty for demos!

### 2. Install and Configure

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and paste your key: GOOGLE_API_KEY=your_key_here
```

## 5 Agentic Patterns

| Pattern | Description | Framework | File |
|---------|-------------|-----------|------|
| **Tool Usage** | Agent uses external tools/functions | LangGraph | `langgraph_01_tools.py` |
| **Planning** | Agent creates plan, executes step-by-step | LangGraph | `langgraph_02_planning.py` |
| **Reflection** | Agent critiques and improves output | LangGraph | `langgraph_03_reflection.py` |
| **Multi-Agent** | Multiple specialized agents collaborate | CrewAI | `crewai_01_multiagent.py` |
| **Memory** | Agent remembers past interactions | CrewAI | `crewai_02_memory.py` |

## Framework Overview

### LangGraph
- Build agents as **graphs** (nodes + edges)
- Great for: stateful workflows, tool usage, conditional logic
- Key concepts: `StateGraph`, nodes, edges, conditional routing

### CrewAI
- Build **teams of agents** that collaborate
- Great for: multi-agent systems, role-based tasks
- Key concepts: `Agent`, `Task`, `Crew`, memory

### n8n
- **Visual** workflow builder (no code)
- Great for: quick prototypes, integrations, demos
- Key concepts: nodes, connections, triggers

## Run Examples

```bash
# LangGraph - Tool Usage
python langgraph_01_tools.py

# LangGraph - Planning  
python langgraph_02_planning.py

# LangGraph - Reflection
python langgraph_03_reflection.py

# CrewAI - Multi-Agent
python crewai_01_multiagent.py

# CrewAI - Memory
python crewai_02_memory.py
```

## Files

```
├── requirements.txt
├── .env.example
├── langgraph_01_tools.py      # Tool Usage Pattern
├── langgraph_02_planning.py   # Planning Pattern
├── langgraph_03_reflection.py # Reflection Pattern
├── crewai_01_multiagent.py    # Multi-Agent Pattern
└── crewai_02_memory.py        # Memory Pattern
```

## Presentation Flow (10 min)

1. **Theory** (3 min): Explain 5 patterns briefly
2. **Frameworks** (2 min): LangGraph, CrewAI, n8n overview
3. **Demos** (5 min):
   - n8n: Reflection pattern (visual)
   - LangGraph: Tool usage OR planning
   - CrewAI: Multi-agent
