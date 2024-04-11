# Prompt Learner

[![Documentation](https://img.shields.io/badge/docs-promptlearner.attuna.xyz-blue.svg)](https://promptlearner.attuna.xyz/)
[![Discord](https://img.shields.io/badge/discord-prompt_learner-blue?logo=discord&logoColor=white&color=5d68e8)](https://discord.gg/FST9HRNKYX)


## What is Prompt Learner?
Prompt Learner is designed to assemble and optimze modular prompts.
A prompt is composed of distinct modules where each module can be optimized both on its own, and as a part of the complete system. Some modules are -

1. The task type
2. The task description
3. A few examples
4. Instructions for output format
5. Custom Prompt Technique specific Instructions



See the documentation on ["Why Prompt Learner?"](https://promptlearner.attuna.xyz/why.html) to learn more.

## Getting started

You can `pip install` Prompt Learner: 

```bash
pip install prompt-learner
```

> [!TIP]
> See the [getting started tutorial](https://promptlearner.attuna.xyz/getting-started.html) for a detailed example of Prompt Learner in action.

## How it works
![Architecture](docs/concepts/images/architecture.png)
Prompt-learner runs on the above architecture.
Starting from the left, the user has to decide on 3 aspects -
1. The Task
2. A set of Examples
3. An LLM Adapter

A task and examples feed into the template of choice (Claude, Open AI..).
The task and examples also interact with selectors which can pick the best n examples for the task using statistical and machine learning techniques.
These selected examples slot into the template, along with any custom instructions from any prompting technique( such as adding 'think step by step' for chain of thought prompting) comprise the final prompt. 
The prompt invokes the LLM through the adapter with any given inference sample to produce the final output.