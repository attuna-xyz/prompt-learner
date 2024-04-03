"""This module contains the OpenAICompletionTemplate class"""
from typing import List
from templates.template import Template
from examples.example import Example


class OpenAICompletionTemplate(Template):
    """This class generates a template for OpenAI completions"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.descriptor = f"""You are a helpful AI assistant.
        You are helping a user with a {self.task_type} task.
        The user asks you to {self.task_description}.
        You have to select from the following labels.
        {self.allowed_labels}."""
        self.prediction_preamble = f"""Given the text, 
        you have to now predict the labels from the 
        list of allowed labels - {self.allowed_labels}."""
        self.examples_preamble = """Here are a few examples to help you
        understand the task better."""
       
    def format_examples(self, examples: List[Example]):
        """Formats the task examples into a string."""
        examples_str = ""
        for example in examples:
            # Assuming 'example' can be directly converted to string.
            examples_str += f"""
            text: {example.text}\n
            label: {example.label}\n"""
        return examples_str

    def add_prediction_sample(self, text: str):
        """Add prediction sample to task."""
        return f"""\n
        text: {text}
        label: """
