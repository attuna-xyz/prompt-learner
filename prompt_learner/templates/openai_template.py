"""This module contains the OpenAICompletionTemplate class"""
from tasks.task import Task


class OpenAICompletionTemplate:
    """This class generates a template for OpenAI completions"""
    def __init__(self, task: Task):
        self.task_description = task.description
        self.task_type = task.__doc__
        self.allowed_labels = task.allowed_labels
        self.template = f"""You are a helpful AI assistant.
        You are helping a user with a {self.task_type} task.
        The user asks you to {self.task_description}.
        You have to select from the following labels.
        {self.allowed_labels}"""
        