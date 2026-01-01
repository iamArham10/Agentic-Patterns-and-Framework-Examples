"""
CREWAI - MULTI-AGENT PATTERN

CrewAI makes it easy to create teams of agents that collaborate.
Each agent has a role, goal, and backstory.
Tasks are assigned to agents and executed in sequence.
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM

load_dotenv()


llm = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY")
)


researcher = Agent(
    role="Researcher",
    goal="Find key facts and information about topics",
    backstory="You are an expert researcher who finds accurate, relevant information.",
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
    backstory="You are a meticulous editor who ensures quality and clarity.",
    llm=llm,
    verbose=True
)


def create_tasks(topic: str):
    """Create tasks for the crew."""
    
    research_task = Task(
        description=f"Research the topic: {topic}. Find 3-4 key facts.",
        expected_output="A list of 3-4 key facts about the topic",
        agent=researcher
    )
    
    writing_task = Task(
        description=f"Write a short article about: {topic}. Use the research provided.",
        expected_output="A 2-3 paragraph article",
        agent=writer
    )
    
    editing_task = Task(
        description="Edit and polish the article. Fix any issues and improve clarity.",
        expected_output="A polished, publication-ready article",
        agent=editor
    )
    
    return [research_task, writing_task, editing_task]


def run_crew(topic: str):
    """Run the multi-agent crew on a topic."""
    print("=" * 50)
    print(f"TOPIC: {topic}")
    print("=" * 50)
    
    tasks = create_tasks(topic)
    
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=tasks,
        verbose=True  # See what each agent does
    )
    
    result = crew.kickoff()
    
    print("\n" + "=" * 50)
    print("FINAL OUTPUT:")
    print("=" * 50)
    print(result)
    
    return result

# RUN

if __name__ == "__main__":
    print("=" * 50)
    print("CREWAI - MULTI-AGENT PATTERN")
    print("=" * 50)
    print("""
Agents: Researcher -> Writer -> Editor
Each agent has a role and works on their task.
Output from one agent feeds into the next.
    """)
    
    run_crew("The benefits of drinking water")
