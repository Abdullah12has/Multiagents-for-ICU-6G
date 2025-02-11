from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
import os

load_dotenv()

@CrewBase
class Icsagents:
    """Icsagents crew for mathematical problem solving"""

    def __init__(self, context_data=None):
        """
        Initialize with configuration files, LLM setup, and context data
        
        Args:
            context_data (dict, optional): Context data to be used by agents and tasks
        """
        self.agents_config = self._load_config('config/agents.yaml')
        self.tasks_config = self._load_config('config/tasks.yaml')
        self.context_data = context_data or {}
        
        self.ollama_llm = LLM(
            model='ollama/tinyllama',
            base_url='http://ollama:11434/api/generate',
        )

    def _load_config(self, path):
        """Helper method to load configuration files"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        # Add your YAML loading logic here
        return {}  # Replace with actual config loading

    @agent
    def math_researcher(self) -> Agent:
        """Agent that performs mathematical research based on the provided context."""
        if not self.agents_config.get('math_researcher'):
            raise ValueError("Math researcher configuration not found")
            
        return Agent(
            role="Mathematical Researcher",
            goal="Analyze and break down mathematical problems",
            backstory="Expert in mathematical research and problem decomposition",
            tools=self.agents_config['math_researcher'].get('tools', []),
            llm=self.ollama_llm,
            verbose=True,
            memory=True,
            context=self.context_data
        )

    @agent
    def math_analyst(self) -> Agent:
        """Agent that analyzes mathematical data with context."""
        if not self.agents_config.get('math_analyst'):
            raise ValueError("Math analyst configuration not found")
            
        return Agent(
            role="Mathematical Analyst",
            goal="Generate detailed mathematical solutions and insights",
            backstory="Expert in mathematical analysis and solution generation",
            tools=self.agents_config['math_analyst'].get('tools', []),
            llm=self.ollama_llm,
            verbose=True,
            memory=True,
            context=self.context_data
        )
@task
def research_task(self, topic: str) -> Task:
    """Task for breaking down mathematical problems."""
    if not self.tasks_config.get('problem_breakdown_task'):
        raise ValueError("Problem breakdown task configuration not found")

    return Task(
        description=f"Research and break down the mathematical topic: {topic}",
        agent=self.math_researcher(), 
        context={"topic": topic, "context": self.context_data}
    )

@task
def reporting_task(self) -> Task:
    """Task for generating mathematical solution reports."""
    if not self.tasks_config.get('solution_generation_task'):
        raise ValueError("Solution generation task configuration not found")

    return Task(
        description="Generate comprehensive mathematical solution report",
        agent=self.math_analyst(),  
        context=self.context_data,
        output_file='report.md'
    )

@crew
def crew(self, topic: str) -> Crew:
    """
    Creates and executes the crew with the provided topic.
    """
    return Crew(
        agents=[
            self.math_researcher(),  
            self.math_analyst() 
        ],
        tasks=[
            self.research_task(topic), 
            self.reporting_task() 
        ],
        process=Process.sequential,
        verbose=True
    )