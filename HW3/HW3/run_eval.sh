#!/bin/sh
export CARLA_ROOT=/media/hcis-s09/pny_2tb/HW3/CARLA_0.9.14_instance_id
export WORK_DIR=/media/hcis-s09/pny_2tb/HW3/Interaction-Generalization

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/team_code
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hcis-s16/miniconda3/lib

level=hard
export EVAL_CONFIG=${WORK_DIR}/eval_config/sensor_eval_${level}.json
export SENSOR_AGNET=${WORK_DIR}/SRL_agent
export AGENT_CONFIG=${WORK_DIR}/checkpoints/BB_BT_R_O_wp
# export AGENT_CONFIG=${WORK_DIR}/checkpoints/PlanT

export DIRECT=0
export RESUME=1
export REND=0
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/${level}/BB_BT_R_O/interaction.json
export SAVE_PATH=${WORK_DIR}/results/${level}/BB_BT_R_O


SERVICE="CarlaUE4"

killall -9 -r CarlaUE4-Linux 
bash ${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen &

sleep 5

while true ; do 
  python multi_agent_eval.py \
  --eval_config ${EVAL_CONFIG} \
  --sensor_agent ${SENSOR_AGNET} \
  --agent_config ${AGENT_CONFIG} \
  --checkpoint ${CHECKPOINT_ENDPOINT} \
  --resume ${RESUME}

  # check if carla be alive
  if pgrep "$SERVICE" >/dev/null
  then
    echo "$SERVICE is running"
    break
  else
    echo "$SERVICE is  stopped"
    bash ${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen & sleep 5
  fi
done

sleep 5

killall -9 -r CarlaUE4-Linux 

sleep 1