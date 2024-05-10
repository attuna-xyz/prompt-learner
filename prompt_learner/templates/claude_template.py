"""This module contains the ClaudeTemplate class"""
from typing import List
from prompt_learner.examples.example import Example
from .template import Template


class ClaudeTemplate(Template):
    """This class generates a template for Anthropic completions"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tasks_with_labels = ["Classification", "Tagging"]
        self.descriptor = f"""You are a helpful AI assistant.  \nYou are helping a user with a {self.task_type} task.  \nThe user gives you the following task description.  \n<task_description>{self.task_description}</task_description>  \n"""
        if self.allowed_labels:
            self.descriptor += f"""You have to select from the following labels.  \n<allowed_labels>{self.allowed_labels}</allowed_labels>"""
        if self.task_type in tasks_with_labels:
            self.prediction_preamble = f"""Given the text, you have to now predict the labels from list of allowed labels - {self.allowed_labels}.  \nOutput only the label(s) and close the <label> tag."""
        elif self.task_type == "SQLGeneration":
            self.prediction_preamble = """Given the text, you have to now generate a SQL query.  \nOnly the SQL query is expected."""
        else:  #generic preamble for prediction
            self.prediction_preamble = """Given the text, you have to now predict."""
        self.examples_preamble = """Here are a few examples to help you understand the task better.  \n"""
    
    def format_examples(self, examples: List[Example]):
        """Formats the task examples into a string."""
        tasks_with_labels = ["Classification", "Tagging"]
        examples_str = ""
        for example in examples:
            if self.task_type in tasks_with_labels:
                examples_str += f"""<example>  \n<text> {example.text}</text>  \n<label> {example.label}</label>  \n</example>  \n"""
            elif self.task_type == "SQLGeneration":
                examples_str += f"""<example>  \n<schema> {example.context}</schema>  \n<text> {example.text}</text>  \n<SQL> {example.label}</SQL>  \n</example>\n"""
            else:
                examples_str += f"""<example>  \n<text> {example.text}</text>  \n<output> {example.label}</output>  \n</example>  \n"""
        return examples_str
    
    def add_prediction_sample(self, text: str, context: str):
        """Add prediction sample to task."""
        tasks_with_labels = ["Classification", "Tagging"]
        prediction_preamble = self.prediction_preamble + f"""  \n <text> {text} <text>"""
        if self.task_type in tasks_with_labels:
            return prediction_preamble + "<label>"
        elif self.task_type == "SQLGeneration":
            return prediction_preamble + f"""  \n <schema>{context} <\schema>\m <SQL>"""
        else:
            return prediction_preamble + "<output>"
