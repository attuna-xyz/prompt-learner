from prompt_learner.translator.translate import Translate
from prompt_learner.tasks.classification import ClassificationTask
from prompt_learner.templates.openai_template import OpenAICompletionTemplate
from prompt_learner.templates.anthropic_template import AnthropicCompletionTemplate
s="""You are going to get an input prompt and you have to extract parts of it.
You are a helpful AI assistant.
You are helping a user with a Classification task.
The user gives you the following task description.
<task_description>This is a task description</task_description>
You have to select from the following labels.
<allowed_labels>["label1","label2"]</allowed_labels>
Here are a few examples to help you understand the task better.
<example>
<text> This is an example text</text>
<label> label1</label>
</example>"""
tr=Translate(input_prompt=s)
tr.translate(ClassificationTask, OpenAICompletionTemplate)
print(tr.prompt)
tr.translate(ClassificationTask, AnthropicCompletionTemplate)
print(tr.prompt)