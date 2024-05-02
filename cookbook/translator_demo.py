from prompt_learner.translator.translate import Translate
from prompt_learner.tasks.classification import ClassificationTask
from prompt_learner.templates.openai_template import OpenAICompletionTemplate
from prompt_learner.templates.anthropic_template import AnthropicCompletionTemplate
s="testing"
tr=Translate(input_prompt=s)
tr.translate(ClassificationTask, OpenAICompletionTemplate)
print(tr.prompt)
tr.translate(ClassificationTask, AnthropicCompletionTemplate)
print(tr.prompt)