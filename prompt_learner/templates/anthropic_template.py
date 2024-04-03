"""This module contains the AnthropicCompletionTemplate class"""
from typing import List
from examples.example import Example
from templates.template import Template


class AnthropicCompletionTemplate(Template):
    """This class generates a template for Anthropic completions"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.descriptor = f"""You are a helpful AI assistant.
        You are helping a user with a {self.task_type} task.
        You have to perform the following task.
        <task_description>{self.task_description}</task_description>
        You have to select from the following labels.
        <allowed_labels>{self.allowed_labels}</allowed_labels>"""
        self.prediction_preamble = f"""Given the text, you have
        to now predict the labels from
        list of allowed labels - {self.allowed_labels}
        Output only the label(s) and close the <label> tag."""
        self.examples_preamble = """ Here are a few examples to
        help you understand the task better."""
    
    def format_examples(self, examples: List[Example]):
        """Formats the task examples into a string."""
        examples_str = ""
        for example in examples:
            examples_str += f"""
            <example>
            <text> {example.text}</text>\n
            <label> {example.label}</label>\n
            </example>"""
        return examples_str
    
    def add_prediction_sample(self, text: str):
        """Add prediction sample to task."""
        return f"""\n
        <text> {text} </text>\n
        """
        