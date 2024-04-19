from prompt_learner.adapters.openai import OpenAI
from prompt_learner.adapters.anthropic import Anthropic
from prompt_learner.adapters.llama import Llama
from prompt_learner.templates.openai_template import OpenAICompletionTemplate
from prompt_learner.templates.anthropic_template import AnthropicCompletionTemplate
from prompt_learner.tasks.sql_generation import SQLGenerationTask


from prompt_learner.examples.example import Example

from prompt_learner.optimizers.selectors.random_sampler import RandomSampler
from prompt_learner.prompts.cot import CoT

sql_description = "Please generate SQL query for the given texts to run on sqlite. I will use your output directly in sqlite so only give me the final executable SQL."
sql_task = SQLGenerationTask(description=sql_description)
schema = """CREATE TABLE singer (
 singer_id NUMERIC PRIMARY KEY,
    name TEXT,
    country TEXT,
    song_name TEXT,
    song_release_year TEXT,
    age NUMERIC,
    is_male TIMESTAMP
);
"""
sql_task.add_example(Example(text="How many singers do we have?", context=schema, label="SELECT COUNT(singer_id) FROM singer;"))
sql_task.add_example(Example(text="What is the average, minimum, and maximum age for all French singers?", context=schema, label="SELECT AVG(age), MIN(age), MAX(age) FROM singer WHERE country='France';"))


task = sql_task
openai_template = OpenAICompletionTemplate(task=sql_task)
sampler = RandomSampler(num_samples=1, task=sql_task)
sampler.select_examples()
openai_prompt = CoT(template=openai_template, selector=sampler)
openai_prompt.assemble_prompt()
openai_prompt.add_inference("Show number of singers in France", schema)
print(openai_prompt.prompt)
print(task.predict(OpenAI(), openai_prompt.prompt))
print(task.predict(Llama(), openai_prompt.prompt))
#using anthropic
anthropic_template = AnthropicCompletionTemplate(task=sql_task)
anthropic_prompt = CoT(template=anthropic_template, selector=sampler)
anthropic_prompt.assemble_prompt()
anthropic_prompt.add_inference("Show number of singers in France", schema)
print(anthropic_prompt.prompt)
print(task.predict(Anthropic(), anthropic_prompt.prompt))