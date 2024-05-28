from prompt_learner.adapters.openai import OpenAI
from prompt_learner.adapters.anthropic import Anthropic
from prompt_learner.adapters.llama import Llama
from prompt_learner.templates.markdown import MarkdownTemplate
from prompt_learner.templates.xml import XmlTemplate
from prompt_learner.tasks.sql_generation import SQLGenerationTask


from prompt_learner.examples.example import Example

from prompt_learner.selectors.random_sampler import RandomSampler
from prompt_learner.prompts.cot import CoT

sql_description = "Please generate SQL query for the given texts to run on sqlite. I will use your output directly in sqlite so only give me the final executable SQL. Do not wrap it in backticks or quotes."
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
markdown_template = MarkdownTemplate(task=sql_task)
sampler = RandomSampler(num_samples=1, task=sql_task)
sampler.select_examples()
gpt_prompt = CoT(template=markdown_template, selector=sampler)
gpt_prompt.assemble_prompt()
gpt_prompt.add_inference("Show number of singers in France", schema)
print(gpt_prompt.prompt)
print(task.predict(OpenAI(model_name='gpt-4-turbo'), gpt_prompt.prompt))
print(task.predict(Llama(), gpt_prompt.prompt))
#using anthropic
claude_template = XmlTemplate(task=sql_task)
claude_prompt = CoT(template=claude_template, selector=sampler)
claude_prompt.assemble_prompt()
claude_prompt.add_inference("Show number of singers in France", schema)
print(claude_prompt.prompt)
print(task.predict(Anthropic(), claude_prompt.prompt))