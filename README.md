# pandas_agent

A python package to work with pandas using LLMs. Example:

```
llm_connector = OpenAIConnector(
    api_key='my-key',
    model='gpt-4o-mini'
)

pandas_agent = PandasAgent(llm_connector=llm_connector, log_responses=True)

pandas_agent.add_task('load the jobs.csv file and show me a list of all columns')
pandas_agent.action()
```