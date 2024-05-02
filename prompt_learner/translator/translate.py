"""A class for a Generic Translation of String Prompt."""
from pydantic import Field, BaseModel
from prompt_learner.templates.template import Template
from prompt_learner.tasks.task import Task
from prompt_learner.examples.example import Example


class Translate(BaseModel):
    """Defines the contract for a Generic Translation."""
    input_prompt: str = Field(description="Input prompt string.", default="")
    task: Task = Field(description="Task type which the string is solving for.",default=None)
    prompt: str = Field(description="Final prompt string.", default="")
    template: Template = Field(description="Template for the prompt.", default=None)
    
    def assemble_prompt(self):
        """Assemble the prompt."""
        self.prompt = f"""{self.template.descriptor}{self.template.examples_preamble}
        {self.template.format_examples(self.template.task.selected_examples)}"""
    
    def create_task(self, task_description: str, allowed_labels: list):
        """Create a task for the prompt."""
        self.task = self.task(description=task_description, allowed_labels=allowed_labels)

    def create_examples(self, examples: list):
        """Create examples for the task."""
        for example in examples:
            self.task.add_example(Example(text=example['text'], label=example['label']))
        

    def extract_modules(self):
        """Extract different modules from the string using LLM."""
        task_description = "This is a task description"
        allowed_labels = ["label1", "label2"]
        examples = [{"text": "This is an example text", "label": "label1"}]
        return {'task_description': task_description, 'allowed_labels': allowed_labels, 'examples': examples}
        
    
    def translate(self, task: Task, template: Template):
        """Translate the prompt to the new template."""
        self.task = task
        all_modules = self.extract_modules()
        self.create_task(all_modules['task_description'], all_modules['allowed_labels'])
        self.create_examples(all_modules['examples'])
        self.template = template(task=self.task)
        self.assemble_prompt()