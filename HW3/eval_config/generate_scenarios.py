import json
from copy import deepcopy

agents = ["auto", "plant", "roach", "sensor"]
eavl_agent = "sensor"

f = open("eval_hard.json")
eval_config = json.load(f)
scenario_list = []
index = 0
for scenario in eval_config['available_scenarios']:
  for agent in agents:
    scenario['ego_agent']['type'] = eavl_agent
    scenario['Index'] = index
    index += 1
    for other_agent in scenario['other_agents']:      
      other_agent['type'] = agent
    temp_scenario = deepcopy(scenario)
    scenario_list.append(temp_scenario)
    if agent == eavl_agent:
      continue

    scenario['ego_agent']['type'] = agent
    scenario['Index'] = index
    index += 1
    for other_agent in scenario['other_agents']:      
      other_agent['type'] = eavl_agent
    temp_scenario = deepcopy(scenario)
    scenario_list.append(temp_scenario)
    
eval_config['available_scenarios'] = scenario_list
with open(f"{eavl_agent}_eval_hard.json", 'w') as fd:
  json.dump(eval_config, fd, indent=2, sort_keys=False)