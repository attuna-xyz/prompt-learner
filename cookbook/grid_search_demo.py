from prompt_learner.optimizers.grid_search import GridSearch
from prompt_learner.tasks.classification import ClassificationTask
from prompt_learner.examples.example import Example
from prompt_learner.templates.markdown import MarkdownTemplate
from prompt_learner.templates.xml import XmlTemplate
from prompt_learner.adapters.ollama_local import OllamaLocal
from prompt_learner.adapters.openai import OpenAI
from prompt_learner.adapters.anthropic import Anthropic
from prompt_learner.selectors.random_sampler import RandomSampler

from prompt_learner.selectors.diverse_sampler import DiverseSampler
from prompt_learner.prompts.cot import CoT

task_description = "You have to classify customer texts as Urgent or Not Urgent"
allowed_labels = ["Urgent", "Not Urgent"]
classification_task = ClassificationTask(description=task_description, allowed_labels=allowed_labels)

template = MarkdownTemplate(task=classification_task)

with open("data/support_texts_test.csv") as f:
    idx=0
    for line in f:
        idx+=1
        text, label = line.split(",")
        classification_task.add_example(Example(text=text.strip(), label=label.strip()))

sampler = RandomSampler(num_samples=2, task=classification_task)
sampler.select_examples()

claude_prompt = CoT(template=template)
claude_prompt.assemble_prompt()

# initialize a grid search on current prompt
grid_search = GridSearch(prompt=claude_prompt)

# no sampler
#param_grid = {'template': [MarkdownTemplate, XmlTemplate], 'adapter': [Anthropic(model_name="claude-3-haiku-20240307"), OpenAI(model_name='gpt-4o')]}

# sampler and local model
#param_grid = {'sampler': [RandomSampler(num_samples=5,task=classification_task)], 'template': [MarkdownTemplate, XmlTemplate], 'adapter': [OllamaLocal()]}

#sampler, and models and templates
param_grid = {'sampler': [RandomSampler(num_samples=5,task=classification_task), DiverseSampler(num_samples=2,task=classification_task)], 'template': [MarkdownTemplate, XmlTemplate], 'adapter': [Anthropic(model_name="claude-3-haiku-20240307"), OpenAI(model_name='gpt-4o')]}
best_params, all_results = grid_search.search(param_grid)

print(best_params)
print(all_results)