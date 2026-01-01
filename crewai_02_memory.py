"""
CREWAI - MEMORY PATTERN

Memory allows agents to remember past interactions and learn.
CrewAI supports different types of memory:
- Short-term: Current conversation context
- Long-term: Persisted across sessions
- Entity: Remember facts about specific entities
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()

# ============================================================
# STEP 1: Configure LLM
# ============================================================

llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# ============================================================
# STEP 2: Create Agent WITH Memory Enabled
# ============================================================

assistant = Agent(
    role="Personal Assistant",
    goal="Help users and remember their preferences",
    backstory="""You are a helpful assistant with excellent memory.
    You remember user preferences and past conversations.
    You use this memory to provide personalized responses.""",
    llm=llm,
    memory=True,  # Enable memory for this agent
    verbose=True
)

# ============================================================
# STEP 3: Create Crew WITH Memory Enabled
# ============================================================

def create_memory_crew():
    """Create a crew with memory capabilities."""
    return Crew(
        agents=[assistant],
        tasks=[],  # We'll add tasks dynamically
        memory=True,           # Enable crew-level memory
        verbose=True
    )

# ============================================================
# STEP 4: Simulate Conversation with Memory
# ============================================================

def chat_with_memory():
    """Demonstrate memory across multiple interactions."""
    
    print("=" * 50)
    print("CREWAI - MEMORY PATTERN DEMO")
    print("=" * 50)
    print("""
This demo shows how agents can remember information
across multiple interactions.
    """)
    
    crew = create_memory_crew()
    
    # Conversation 1: Tell the agent something
    print("\n--- INTERACTION 1 ---")
    task1 = Task(
        description="The user says: 'My name is Alex and I love Python programming.'",
        expected_output="Acknowledge and remember this information",
        agent=assistant
    )
    crew.tasks = [task1]
    result1 = crew.kickoff()
    print(f"Response: {result1}\n")
    
    # Conversation 2: Ask something that requires memory
    print("\n--- INTERACTION 2 ---")
    task2 = Task(
        description="The user asks: 'What programming language do I like?'",
        expected_output="Recall what you remember about the user",
        agent=assistant
    )
    crew.tasks = [task2]
    result2 = crew.kickoff()
    print(f"Response: {result2}\n")
    
    # Conversation 3: Add more info and test recall
    print("\n--- INTERACTION 3 ---")
    task3 = Task(
        description="The user says: 'I also enjoy machine learning projects.'",
        expected_output="Acknowledge and add to what you know about the user",
        agent=assistant
    )
    crew.tasks = [task3]
    result3 = crew.kickoff()
    print(f"Response: {result3}\n")
    
    # Conversation 4: Test combined memory
    print("\n--- INTERACTION 4 ---")
    task4 = Task(
        description="The user asks: 'Based on what you know about me, suggest a project I might enjoy.'",
        expected_output="Give a personalized suggestion based on remembered preferences",
        agent=assistant
    )
    crew.tasks = [task4]
    result4 = crew.kickoff()
    print(f"Response: {result4}\n")
    
    print("=" * 50)
    print("KEY TAKEAWAY:")
    print("=" * 50)
    print("""
The agent remembered:
1. User's name (Alex)
2. Favorite language (Python)
3. Interest in ML

This memory persists across interactions!
    """)

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    chat_with_memory()
