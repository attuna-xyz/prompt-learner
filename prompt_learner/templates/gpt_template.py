"""This module contains the GPTTemplate class"""
from typing import List
from prompt_learner.examples.example import Example
from .template import Template


class GPTTemplate(Template):
    """This class generates a template for OpenAI completions"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tasks_with_labels = ["Classification", "Tagging"]
        self.descriptor = f"""You are a helpful AI assistant.  \nYou are helping a user with a {self.task_type} task.  \nThe user gives you the following task description.  \n{self.task_description}\n"""
        if self.allowed_labels:
            self.descriptor += f"""You have to select from the following labels.  \n{self.allowed_labels}.  \nOnly output labels and nothing else"""
        if self.task_type in tasks_with_labels:
            self.prediction_preamble = f"""Given the text, you have to now predict the labels from the list of allowed labels - {self.allowed_labels}."""
        elif self.task_type == "SQLGeneration":
            self.prediction_preamble = """Given the text, you have to now generate a SQL query.Only output the SQL and nothing else."""
        else:  #generic preamble for prediction
            self.prediction_preamble = """Given the text, you have to now predict."""
        self.examples_preamble = """Here are a few examples to help you understand the task better.  \n"""
       
    def format_examples(self, examples: List[Example]):
        """Formats the task examples into a string."""
        tasks_with_labels = ["Classification", "Tagging"]
        examples_str = ""
        for example in examples:
            if self.task_type in tasks_with_labels:
                examples_str += f"""text: {example.text}  \nlabel: {example.label}  \n"""
            elif self.task_type == "SQLGeneration":
                examples_str += f"""schema: {example.context}  \ntext: {example.text}  \nSQL: {example.label}  \n"""
            else: #generic example format
                examples_str += f"""
                text: {example.text}  \noutput: {example.label}  \n"""
        return examples_str

    def add_prediction_sample(self, text: str, context: str = None):
        """Add prediction sample to task."""
        tasks_with_labels = ["Classification", "Tagging"]
        if self.task_type in tasks_with_labels:
            return self.prediction_preamble + f"""  \ntext: {text}"""+ "  \nlabel:"
        elif self.task_type == "SQLGeneration":
            return self.prediction_preamble + f"""  \nschema: {context}  \ntext: {text}  \nSQL: """
        else:
            return self.prediction_preamble + "  \noutput:"
