from prompt_learner.tasks.classification import ClassificationTask
from prompt_learner.examples.example import Example
from prompt_learner.prompts.prompt import Prompt
from prompt_learner.prompts.cot import CoT
from prompt_learner.templates.gpt_template import GPTTemplate
from prompt_learner.adapters.openai import OpenAI
from prompt_learner.adapters.llama  import Llama
from prompt_learner.selectors.random_sampler import RandomSampler
from prompt_learner.selectors.diverse_sampler import DiverseSampler
from prompt_learner.evals.metrics.accuracy import Accuracy

task_description = "You have to classify customer texts as Urgent or Not Urgent"
allowed_labels = ["Urgent", "Not Urgent"]
classification_task = ClassificationTask(description=task_description, allowed_labels=allowed_labels)

template = GPTTemplate(task=classification_task)

classification_task.add_example(Example(text="Could you provide me with more information about your refund policy?", label="Not Urgent"))
classification_task.add_example(Example(text="My service is down and I am losing sales because of this!", label="Urgent"))
classification_task.add_example(Example(text="I am facing a deadline in 1 hour and cannot access my account", label="Urgent"))
classification_task.add_example(Example(text="Could you guide me to the recent changes in the pricing structure?", label="Not Urgent"))

sampler = RandomSampler(num_samples=2, task=classification_task)
sampler.select_examples()

gpt_prompt = CoT(template=template)
gpt_prompt.assemble_prompt()
print(gpt_prompt.prompt)


with open("data/support_texts_test.csv") as f:
    idx=0
    for line in f:
        print(idx,line)
        idx+=1
        text, label = line.split(",")
        classification_task.add_example(Example(text=text.strip(), label=label.strip()), test=True)





acc, results = Accuracy(classification_task).compute(gpt_prompt, Llama(),test=True)
print(results)
print("got a test accuracy of ", acc, " with ", len(results), " eval samples")



classification_task.add_example(Example(text="I need to update my payment method before my subscription renews tonight.", label="Urgent"))
classification_task.add_example(Example(text="Could you please update me on the status of when you are releasing the AI enabled features?", label="Not Urgent"))

sampler = DiverseSampler(num_samples=3, task=classification_task)
sampler.select_examples()

gpt_prompt = CoT(template=template, selector=sampler)
gpt_prompt.assemble_prompt()

print(gpt_prompt.prompt)

acc,results = Accuracy(classification_task).compute(gpt_prompt, Llama(),test=True)
print("got a test accuracy of ", acc, " with ", len(results), " eval samples")




gpt_prompt.add_inference("You guys are the best! I love your service!")
answer = classification_task.predict(OpenAI(), gpt_prompt.prompt)
print(answer)