"""Compute accuracy metric for the prompt+model."""
from copy import deepcopy
from prompt_learner.tasks.task import Task
from prompt_learner.prompts.prompt import Prompt
from prompt_learner.adapters.adapter import Adapter


class Accuracy:
    """Defines the contract for the Accuracy Metric."""
    def __init__(self, task: Task):
        self.task = task

    def compute(self, prompt: Prompt, adapter: Adapter):
        """Compute the accuracy of the model."""
        correct = 0
        total = 0
        for example in self.task.examples:
            if example not in self.task.selected_examples:
                total += 1
                temp_prompt = deepcopy(prompt)
                temp_prompt.add_inference(example.text)
                prediction = self.task.predict(adapter, temp_prompt.prompt)
                if example.label == prediction:
                    correct += 1
        return correct / total
