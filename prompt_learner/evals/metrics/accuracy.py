"""Compute accuracy metric for the prompt+model."""
from copy import deepcopy
from prompt_learner.tasks.task import Task
from prompt_learner.prompts.prompt import Prompt
from prompt_learner.adapters.adapter import Adapter

class Accuracy:
    """Defines the contract for the Accuracy Metric."""
    def __init__(self, task: Task):
        self.task = task

    def compute(self, prompt: Prompt, adapter: Adapter, test=False) -> tuple[float, int]:
        """Compute the accuracy of the model."""
        correct = 0
        total = 0
        if test is False:
            examples = self.task.examples
        else:
            examples = self.task.test_examples
        for example in examples:
            if example not in self.task.selected_examples:
                total += 1
                temp_prompt = deepcopy(prompt)
                temp_prompt.add_inference(example.text)
                prediction = self.task.predict(adapter, temp_prompt.prompt)
                prediction = prediction.strip()
                prediction = prediction.strip('\n')
                if example.label == prediction:
                    correct += 1
                else:
                    print("Incorrect prediction:", example.text, example.label,prediction)
        return (correct / total, total)
