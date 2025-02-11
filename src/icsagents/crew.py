from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

load_dotenv()
# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


file_path = "./src/icsagents/problems.txt"

# Read the file and store its contents in a variable
with open(file_path, "r", encoding="utf-8") as file:
	content = file.read()
print(content)


@CrewBase
class Icsagents():
	"""Icsagents crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	ollama_llm = LLM(
		model = 'ollama/deepseek-r1:14b',
		base_url = 'http://localhost:11434',
		# base_url = 'http://ollama:11434/api/generate',
        
	)
	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def math_researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['math_researcher'],
			llm=self.ollama_llm,
			verbose=True
		)

	@agent
	def math_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['math_analyst'],
			llm=self.ollama_llm,
			verbose=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['problem_breakdown_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['solution_generation_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Icsagents crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge
		string_source = StringKnowledgeSource(content=content)
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			knowledge_sources=[string_source],
			# memory=True
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
