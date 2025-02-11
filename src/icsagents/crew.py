from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

load_dotenv()

@CrewBase
class Icsagents():
    """Icsagents crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    ollama_llm = LLM(
        model='ollama/deepseek-r1:14b',
        base_url='http://ollama:11434/api/generate',
    )

    @agent
    def math_researcher(self, context_data) -> Agent:
        """Agent that performs research based on the provided context."""
        return Agent(
            config=self.agents_config['math_researcher'],
            llm=self.ollama_llm,
            verbose=True,
            memory=True,
            additional_context={"context": context_data}  # Pass context dynamically
        )

    @agent
    def math_analyst(self, context_data) -> Agent:
        """Agent that analyzes mathematical data with context."""
        return Agent(
            config=self.agents_config['math_analyst'],
            llm=self.ollama_llm,
            verbose=True,
            memory=True,
            additional_context={"context": context_data}  # Pass context dynamically
        )

    @task
    def research_task(self, topic, context_data) -> Task:
        """Task for breaking down problems with topic and context."""
        return Task(
            config=self.tasks_config['problem_breakdown_task'],
            input_data={"topic": topic, "context": context_data}  # Pass topic & context
        )

    @task
    def reporting_task(self, context_data) -> Task:
        """Task for generating reports using provided context."""
        return Task(
            config=self.tasks_config['solution_generation_task'],
            input_data={"context": context_data},
            output_file='report.md'
        )

    @crew
    def crew(self, topic, context_data) -> Crew:
        """Creates and executes the crew with provided topic and context."""
        return Crew(
            agents=[
                self.math_researcher(context_data),
                self.math_analyst(context_data)
            ],
            tasks=[
                self.research_task(topic, context_data),
                self.reporting_task(context_data)
            ],
            process=Process.sequential,
            verbose=True
        )
