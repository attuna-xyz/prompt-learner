"""This module contains the AnthropicCompletionTemplate class"""
from tasks.task import Task


class AnthropicCompletionTemplate:
    """This class generates a template for Anthropic completions"""
    def __init__(self, task: Task):
        self.task_description = task.description
        self.task_type = task.__doc__
        self.allowed_labels = task.allowed_labels
        self.template = f"""You are a helpful AI assistant.
        You are helping a user with a {self.task_type} task.
        You have to perform the following task.
        <task_description>{self.task_description}</task_description>
        You have to select from the following labels.
        <allowed_labels>{self.allowed_labels}</allowed_labels>
        """
        