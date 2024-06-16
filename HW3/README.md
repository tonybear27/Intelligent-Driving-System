# Interaction Benchmark

## Install the Environment

``` bash
conda env create -f environment.yml
```

## Run Evaluation
If your model is developed based on TF++, you should be able to use [SRL_agent.py](./SRL_agent.py) directly or with a few modifications(run_step function at line:275) as the interface between your model and the benchmark. 

For those who started from PlanT, the interface would be [plant_agent](./plant_agent.py) and you might also need to do some small modification to the run_step function at line:130 based on how you develop your model. Additionally, you need to generate new evaluation config by changing the `eval_agent` in [generate_scenarios.py](./eval_config/generate_scenarios.py) to `plant` and run it to generate corresponding evaluation config.

If you started with TCP, you will need to write a new agent file similar to [SRL_agent.py](./SRL_agent.py) and the [longest6_agent](https://github.com/HCIS-Lab/IDS_s24/blob/main/HW0/agents/TCP/longest6_agent.py) from HW0. To run the evaluation of TCP-like agent, plaese change `SENSOR_AGNET` in [run_eval.sh](./run_eval.sh) to the agent file you write.

1. ```bash
   bash run_eval.sh [CARLA_14_ROOT] [WORK_DIR] 
   ```
2. If you want to change the scenario, please modify `EVAL_CONFIG`
  
    a. Basic -> sensor_eval_easy.json
    
    b. In-sequence -> sensor_eval_mid.json
    
    c. Multiple -> sensor_eval_hard.json
