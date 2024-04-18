
**[Opening Scene]**

- Background Music Fades In
- Title Screen: "Why Prompt-Learner?"

**Narrator (Voice Over):** "Welcome to our quick guide on 'Prompt-Learner,' a tool designed to standardize, modularize and optimize how we work with prompts in large language models. If you've ever found yourself struggling to craft and experiment with prompts, then Prompt-Learner is here to simplify your life."

**[Transition to Explanation]**

**Narrator (Voice Over):** "As practitioners and researchers, we often face the challenge of creating and maintaining prompt strings from scratch. With Prompt-Learner, we break down complex prompts into manageable, modular parts. This allows for systematic optimization and greater understanding, making your prompting experience both maintainable and efficient."

**[Display Image on Screen]**

**Narrator (Voice Over):** "Imagine your task as a set of interconnected modules, each with its specific role. At the core, you have the Task module. '
this is where you define your task - by specifying its type (such as classification or tagging) and
giving a short description.

**[Highlight 'Classification, tagging' on Image]**

classification_description = "You have to classify customer texts as Urgent or Not Urgent"


**Narrator (Voice Over):** "Language Models are notoriously chatty and inconsistent in what they spit out. This is why akin to a traditional ML task, you should define the set of output you expect the model provide. In a classification or tagging task, these will be the set of permissable labels.

This helps to not only give clear instructions to the Language model but also ties in with inbuilt Pydantic validations in Prompt Learner to enforce data validation constraints. For example, any label beyond these set of specified labels will be considered invalid ay any step.

**Narrator (Voice Over):** "Next, we specify the labels allowed in this classification."

classification_labels = ["Urgent", "Not Urgent"]


**Narrator (Voice Over):** "We then create the classification task using these parameters."

from prompt_learner.tasks.classification import ClassificationTask
classification_task = ClassificationTask(description=classification_description, allowed_labels=classification_labels)

**[Highlight 'allowed_labels' on Image]**

**Narrator (Voice Over):** Every single Language Model was trained in a very specific way and responds differently to different formatting patterns. For example, Anthropic recommends using xml tags when using Claude. However, xml tags can even be useful in OpenAI's GPT set of models even though they are not strictly recommended because they help in giving clarity to different sections. You can mix and match these templates using prompt learner that ensures recommended formatting.  Lets use the Open AI template for now. 

from prompt_learner.templates.openai_template import OpenAICompletionTemplate
template = OpenAICompletionTemplate(task=classification_task)

**Narrator (Voice Over):** "That completes the task overview.
 Examples are the backbone of any classification task. Prompt-Learner allows you to integrate examples seamlessly to train your model more effectively. 
 It also has a growing list of simple and advanced selectors that help you sample the best n examples from your set of examples to be inserted into the prompt. 
 You can also tie this in with a online data source, for example, production data so that your prompt is always up to date and current with your data distribution.
**Narrator (Voice Over):** "Adding real-world examples to our task is crucial for training. Here, we add two samples."


from prompt_learner.examples.example import Example
classification_task.add_example(Example(text="I need help", label="Urgent"))
classification_task.add_example(Example(text="I got my package", label="Not Urgent"))
selector= RandomSampler(num_samples=1, task=classification_task).select_examples()

**[Highlight 'example_1', 'example_n' on Image]**



**Narrator (Voice Over):** "Custom instructions is another very important area of a prompt. You can implement standardized prompting techniques like chain of thought prompting by adding  'think step by step,'  as a custom instruction or - any of the up and coming sophisticated prompt instruction techniques. All of this contributes to enhancing your prompt's performance."
openai_prompt = CoT(template=template, selector=selector )
**[Highlight 'Custom Instructions' on Image]**

**Narrator (Voice Over):**  Now is a good time to assemble the prompt by combining all the modules we have defined till now.  We have a well formed prompt designed from first principle. We can swap in and swap out any of the intermediate modules and re-assemble the prompt. We are no longer working with a monolithic blob of string.

openai_prompt.assemble_prompt()
openai_prompt.prompt

**Narrator (Voice Over):** Prompt learner has in built validations that can run your assembled prompt against held out data or a custom test dataset. This enables rapid feedback. By changing modules, and re-running evals you can quickly iterate on your prompts. Lets evaluate our current prompt for accuraacy on a few test samples

Sweet we got xxx acc
But I think we can do better.
Lets try adding more examples to training, and using a more sophisticated Sampler. DiverseSampler picks the n most diverse examples so that your training data is better represented in the prompt. 

--
lets re run the eval on unseen test data

et voila better results!
Rapid iteration and experimentation by changing modules of a prompt - thats what we want. Not every change will result in better performance, but we want that feedback and change to happen faster. Prompt learner solves this.

**Narrator (Voice Over):** "You can also choose an LLM provider and run inference using this assmebled and optimized prompt. Given an inference sample, Prompt-Learner will predict the label based on your meticulously crafted prompt."

openai_prompt.add_inference("My package is missing")
answer = classification_task.predict(OpenAI(), openai_prompt.prompt)
print(answer)
**[Highlight the Prediction Module on Image]**

**Narrator (Voice Over):** Give prompt-learner a spin. Check out our docs at promptlearner.attuna.xyz. start us on github, and contribute to the open source project! We are just getting started.

-



---


Hey guys, what's up?  welcome to the quick guide on prompt learner. This is a tool that's designed to modularize and optimize how we work with prompts in large language models.
If you've ever found yourself struggling to craft an experiment with prompts, then you need prompt learner. It breaks down complex prompts into manageable modular parts and allows you to tune & experiment programatically, iteratively and more efficiently.


Your prompt as a set of interconnected modules, each with its own specific role.
Lets begin with the task module. This is where you define your task by specifying its types such as classification, tagging, or any kind of task you're performing, a short description, and some labels.


Let's dive into some code as we talk about it more. I've already imported prompt learner with a few modules and I also have my open AI API key set up.
I have the following task description. Classify customer text as urgent or not urgent. the labels that are allowed are urgent and not urgent, and I define my classification task with these parameters.
Now, language more models are notoriously chatty and inconsistent in what they spit out. This is why, similar to a traditional ML task, you want to define the output that you want, and that's what allowed labels does.
With inbuilt pedantic validations, prompt learner takes care to constrain the output to these labels, otherwise it will be rendered an invalid output.


Every language model was trained in a very specific way that corresponds to different formatting.
For example, entropic recommends using XML tags while using plot. However, OpenAI doesn't recommend this, but you never know, it might help you, it might harm you.
So you want to mix and match these templates and which is why we have the template module in from Plurner that takes care of this.
Lets go with the Anthropic's  template which adheres to guidelines given by Anthropic and which we have set up in the library.


The most important part of a prompt is the few short examples that you give it. And the selection of these examples is even more important. Prompt learner has a growing list of simple and advanced selectors that help you sample the best n examples from your set of examples to be inserted into the prompt.
First, let us add some examples to our task. I will add like four examples for this use case.
You can take your time and read through this, but these are pretty general. 
Lets use a trivial random sampler for now

You can imagine tying this to an online data source, for example, production data so that your prompt is always up-to-date and current with your data distribution.



Lets move on to the custom instructions.
You can use standardize prompting techniques like chain of thought prompting by adding thing step-by-step as a custom instruction or any of the up-and-coming sophisticated prompt instruction techniques.
For now, let us choose chain of thought prompting. 

Lets see how the prompt has been shaping up by assembling. This is the part where everything comes together and you are able to see the current prompt.

This is great because we landed up into a prompt that was built systematically understanding the different modules involved.
And now we can evaluate this prompt against test data. I have a bunch of test samples that I have loaded up in a those examples as test examples in my prompt learner module.


I can run this prompt against these test examples and see how well it performs. We get an accuracy of xx% which is not bad, but I think we can do better.

Lets try adding more examples, and using a more sophisticated DiverseSampler that picks the n most diverse examples so that your training data is better represented in the prompt.
Lets re-run the eval on unseen test data and see how well we perform now.


Wow, we have a better result. Rapid iteration and experimentation by changing modules of a prompt, that's what we want. Not every change will result in better performance, but we want that feedback and change to happen faster. Prompt learner solves this.
Thanks for watching this quick guide on prompt learner. Give it a spin, check out our docs at promptlearner.attuna.xyz, start us on github, and contribute to the open source project. We are just getting started.
