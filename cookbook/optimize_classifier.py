from prompt_learner.adapters.openai import OpenAI

from prompt_learner.templates.gpt_template import GPTTemplate

from prompt_learner.tasks.classification import ClassificationTask

from prompt_learner.examples.example import Example

from prompt_learner.selectors.random_sampler import RandomSampler
from prompt_learner.selectors.stratified_sampler import StratifiedSampler
from prompt_learner.selectors.diverse_sampler import DiverseSampler
from prompt_learner.prompts.cot import CoT
from prompt_learner.prompts.prompt import Prompt #can call it base prompt

classification_description = "You have to classify customer texts as Urgent or Not Urgent"
classification_labels = ["Urgent", "Not Urgent"]
classification_task = ClassificationTask(description=classification_description, allowed_labels=classification_labels)
with open("data/support_texts.csv") as f:
    idx=0
    for line in f:
        print(idx,line)
        idx+=1
        text, label = line.split(",")
        classification_task.add_example(Example(text=text.strip(), label=label.strip()))
with open("data/support_texts_test.csv") as f:
    idx=0
    for line in f:
        print(idx,line)
        idx+=1
        text, label = line.split(",")
        classification_task.add_example(Example(text=text.strip(), label=label.strip()), test=True)

task = classification_task
gpt_template = GPTTemplate(task=classification_task)
sampler = StratifiedSampler(num_samples=2, task=classification_task)
sampler.select_examples()
gpt_prompt = CoT(template=gpt_template)
gpt_prompt.assemble_prompt()
print(gpt_prompt.prompt)
print("Evals,")
acc, num_total_samplers = Accuracy(classification_task).compute(gpt_prompt, OpenAI())
print("got a val accuracy of ", acc, " with ", num_total_samplers, " eval samples")
acc, num_total_samplers = Accuracy(classification_task).compute(gpt_prompt, OpenAI(),test=True)
print("got a test accuracy of ", acc, " with ", num_total_samplers, " eval samples")
sampler = DiverseSampler(num_samples=4, task=classification_task)
sampler.select_examples()
gpt_prompt = CoT(template=gpt_template)
gpt_prompt.assemble_prompt()
print(gpt_prompt.prompt)
print("Evals,")
acc, num_total_samplers = Accuracy(classification_task).compute(gpt_prompt, OpenAI())
print("got a val accuracy of ", acc, " with ", num_total_samplers, " eval samples")
acc, num_total_samplers = Accuracy(classification_task).compute(gpt_prompt, OpenAI(),test=True)
print("got a test accuracy of ", acc, " with ", num_total_samplers, " eval samples")
# gpt_prompt.add_inference("My package is missing")
# print(gpt_prompt.prompt)
# answer = classification_task.predict(OpenAI(), gpt_prompt.prompt)
# print("ANSWER:", answer)

#--

# claude_template = ClaudeTemplate(classification_task,sampler)

# print(claude_template.prompt)
# final_prompt = gpt_template.add_prediction_sample("My package is missing")
# answer = classification_task.predict(OpenAI(), final_prompt)
# print("ANSWER:", answer)
# print("Valid output>", classification_task.validate_prediction("My package is missing", answer))
# final_prompt = claude_template.add_prediction_sample("My package is missing")
# answer = classification_task.predict(Anthropic(), final_prompt)
# print("ANSWER:", answer)
# print("Valid output>", classification_task.validate_prediction("My package is missing", answer))
