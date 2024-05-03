from prompt_learner.translator.translate import Translate
from prompt_learner.tasks.classification import ClassificationTask
from prompt_learner.templates.openai_template import OpenAICompletionTemplate
from prompt_learner.templates.anthropic_template import AnthropicCompletionTemplate
easy="""You are going to get an input prompt and you have to extract parts of it.
You are a helpful AI assistant.
You are helping a user with a Classification task.
The user gives you the following task description.
<task_description>This is a task description</task_description>
"""
difficult="""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

The news items can be categories like below.

Politics, Business, Technology, Health, Environment, Sports, Entertainment, Science, Education, Travel

Below is some example classifications. the format is the news item followed by classification tag.

1. The upcoming federal election is seeing a shift in voter sentiment as recent polls suggest that the incumbent party might lose majority in the parliament. The opposition party, bolstered by this change, has started aggressive campaigning focusing on issues such as healthcare reform and climate change. <classification>Politics</classification>

2. MegaTech Corporation announced a surprising decrease in its quarterly profits, citing supply chain disruptions due to recent global events. The news has resulted in a significant dip in the company's stock prices, and investors are worried about the impact on their portfolios. <classification>Business</classification>

"""
tr=Translate(input_prompt=difficult)
tr.translate(ClassificationTask, OpenAICompletionTemplate)
print(tr.prompt)
tr.translate(ClassificationTask, AnthropicCompletionTemplate)
print(tr.prompt)