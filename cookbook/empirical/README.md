# Example with Empirical

This example shows a side-by-side comparison (with [Empirical](https://github.com/empirical-run/empirical/))
between results from a basic prompt running on OpenAI against an optimized prompt
created via with Prompt Learner.

## Usage

1. Review the Empirical configuration file, which has 2 runs: bare LLM call and a Python
   script (in `main.py`).
   ```
   cat empiricalrc.json
   ```

2. Run inference with the configured models. We will pass the python path from Poetry to run
   the Python script (`main.py`).
   ```
   npx @empiricalrun/cli run --python-path `poetry env info -e`
   ```

3. See results in the Empirical web reporter.
   ```
   npx @empiricalrun/cli ui
   ```

In the web UI, you can change the base prompt for the `main.py` script. Click on "Show config", make a
prompt change, and re-run the column.
