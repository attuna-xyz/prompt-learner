"""This module contains the AnthropicCompletionTemplate class"""
from tasks.task import Task
from templates.template import Template


class AnthropicCompletionTemplate(Template):
    """This class generates a template for Anthropic completions"""
    def __init__(self, task: Task):
        super().__init__(task)
        self.prompt = f"""You are a helpful AI assistant.
        You are helping a user with a {self.task_type} task.
        You have to perform the following task.
        <task_description>{self.task_description}</task_description>
        You have to select from the following labels.
        <allowed_labels>{self.allowed_labels}</allowed_labels>
        Here are a few examples to help you understand the task better.
        <{self.format_examples()}.
        Given the text, you have to now predict the labels from
        list of allowed labels - {self.allowed_labels}
        """
    
    def format_examples(self):
        """Formats the task examples into a string."""
        examples_str = ""
        for example in self.examples:
            examples_str += f"""
            <example>
            <text> {example.text}</text>\n
            <label> {example.label}</label>\n
            </example>"""
        return examples_str
        