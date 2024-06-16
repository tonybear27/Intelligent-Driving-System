#!/bin/sh
export CARLA_ROOT=${1:-/home/hcis-s15/Documents/projects/RiskBench/CARLA_0.9.14_instance_id}
export WORK_DIR=${2:-/home/hcis-s15/Documents/projects/Interaction_Genearlization}

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/team_code

export EVAL_CONFIG=${WORK_DIR}/eval_config/eval_hard.json
export SENSOR_AGNET=${WORK_DIR}/SRL_agent
export AGENT_CONFIG=${WORK_DIR}/checkpoints/BB_BT_R_wp
export DIRECT=0
export RESUME=1
export REND=1
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/eval_hard/interaction.json
export SAVE_PATH=${WORK_DIR}/results/eval_hard


killall -9 -r CarlaUE4-Linux 
bash ${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen &

sleep 5

python multi_agent_eval.py \
--eval_config ${EVAL_CONFIG} \
--sensor_agent ${SENSOR_AGNET} \
--agent_config ${AGENT_CONFIG} \
--checkpoint ${CHECKPOINT_ENDPOINT} \
--resume ${RESUME}

sleep 5

killall -9 -r CarlaUE4-Linux 

sleep 1
