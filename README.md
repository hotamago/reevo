# How to run with langflow
## Install container for docker

Download image
```
docker pull langflowai/langflow:latest
```

Install container
```
docker create --name langflow-reevo -p 7860:7860 langflowai/langflow:latest
```

Import demo flow graph using import function in langflow UI and import `langflow_json\TinhToanTienHoaDemo.json` file

## Connect and run

Clone this repo
```
git clone https://github.com/hotamago/reevo.git
```

Example command
```
python main.py problem=tsp_aco init_pop_size=30 pop_size=10 max_fe=100 timeout=10 algorithm=hota llm_client=langflow llm_client.api_url="http://localhost:7860" llm_client.flow_id="d240dea0-b62d-488b-8866-4a1d13c7d9d7"
```

Explain:
- problem = name of problem to solve (check name in problems folder)
- init_pop_size = initial population size
- pop_size = population size
- max_fe = maximum function evaluations
- timeout = timeout in seconds
- algorithm = algorithm to use (e.g., hota, reevo)
- llm_client = language model client (e.g., langflow, openai)
- llm_client.api_url = API URL of langflow
- llm_client.flow_id = flow ID in langflow
