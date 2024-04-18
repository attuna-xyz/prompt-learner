"""This module contains the OpenAICompletionTemplate class"""
from typing import List
from prompt_learner.examples.example import Example
from .template import Template


class OpenAICompletionTemplate(Template):
    """This class generates a template for OpenAI completions"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tasks_with_labels = ["Classification", "Tagging"]
        self.descriptor = f"""You are a helpful AI assistant.
        You are helping a user with a {self.task_type} task.
        The user asks you to {self.task_description}."""
        if self.allowed_labels:
            self.descriptor += """You have to select from the following labels.
            {self.allowed_labels}."""
        if self.task_type in tasks_with_labels:
            self.prediction_preamble = f"""Given the text, 
            you have to now predict the labels from the 
            list of allowed labels - {self.allowed_labels}."""
        elif self.task_type == "SQLGeneration":
            self.prediction_preamble = """Given the text, 
            you have to now generate a SQL query."""
        else:  #generic preamble for prediction
            self.prediction_preamble = """Given the text, 
            you have to now predict."""
        self.examples_preamble = """Here are a few examples to help you
        understand the task better."""
       
    def format_examples(self, examples: List[Example]):
        """Formats the task examples into a string."""
        tasks_with_labels = ["Classification", "Tagging"]
        examples_str = ""
        for example in examples:
            if self.task_type in tasks_with_labels:
                examples_str += f"""
                text: {example.text}\n
                label: {example.label}\n"""
            elif self.task_type == "SQLGeneration":
                examples_str += f"""
                schema: {example.context}\n
                text: {example.text}\n
                SQL: {example.label}\n"""
            else: #generic example format
                examples_str += f"""
                text: {example.text}\n
                output: {example.label}\n"""
        return examples_str

    def add_prediction_sample(self, text: str, context: str = None):
        """Add prediction sample to task."""
        tasks_with_labels = ["Classification", "Tagging"]
        prediction_preamble = self.prediction_preamble + f"""\n text: {text}"""
        if self.task_type in tasks_with_labels:
            return prediction_preamble + "\n label:"
        elif self.task_type == "SQLGeneration":
            return prediction_preamble + f"""\n schema: {context}\n SQL: """
        else:
            return prediction_preamble + "\n output:"
