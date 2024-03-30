"""This module contains the OpenAICompletionTemplate class"""
from tasks.task import Task


class OpenAICompletionTemplate:
    """This class generates a template for OpenAI completions"""
    def __init__(self, task: Task):
        self.task_description = task.description
        self.task_type = task.__doc__
        self.allowed_labels = task.allowed_labels
        self.examples = task.examples
        self.template = f"""You are a helpful AI assistant.
        You are helping a user with a {self.task_type} task.
        The user asks you to {self.task_description}.
        You have to select from the following labels.
        {self.allowed_labels}.
        Here are a few examples to help you understand the task better.
        {self.format_examples()}
        """
    
    def format_examples(self):
        """Formats the task examples into a string."""
        examples_str = ""
        for example in self.examples:
            # Assuming 'example' can be directly converted to string.
            examples_str += f"""
            text: {example.text}\n
            label: {example.label}\n"""
        return examples_str
