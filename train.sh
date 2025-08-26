# !/bin/bash

exec &> /dev/null

GPU_LIST=(0 1 2 3)


seed_list=(
    0
    1
    2
    3
    4
)


env_list=(
    "hopper-medium-v2"
    "halfcheetah-medium-v2"
    "walker2d-medium-v2"
    "hopper-medium-replay-v2"
    "halfcheetah-medium-replay-v2"
    "walker2d-medium-replay-v2"
    "hopper-medium-expert-v2"
    "halfcheetah-medium-expert-v2"
    "walker2d-medium-expert-v2"
    "antmaze-umaze-v2"
    "antmaze-umaze-diverse-v2"
    "antmaze-medium-play-v2"
    "antmaze-medium-diverse-v2"
    "antmaze-large-play-v2"
    "antmaze-large-diverse-v2"
)


task=0

check_gpu_availability() {
    local gpu_device=$1
    local memory_threshold_percentage=25  # Set your desired memory threshold percentage

    local gpu_info_total=$(nvidia-smi --id=$gpu_device --query-gpu=memory.total --format=csv,noheader,nounits)
    local gpu_info_used=$(nvidia-smi --id=$gpu_device --query-gpu=memory.used --format=csv,noheader,nounits)
    local total_memory=$(echo $gpu_info_total | awk -F ',' '{print $1}')
    local used_memory=$(echo $gpu_info_used | awk -F ',' '{print $1}')
    local memory_threshold=$((total_memory * memory_threshold_percentage / 100))

    if [[ $((total_memory - used_memory)) -ge $memory_threshold ]]; then
        return 0  # GPU has sufficient memory available
    else
        return 1  # GPU does not have sufficient memory available
    fi
}


for env in "${env_list[@]}"; do
  for seed in "${seed_list[@]}"; do
    GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}

    while ! check_gpu_availability $GPU_DEVICE; do
      sleep 60
      let "task=task+1"
      GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
    done

    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python cql_query.py --env $env --seed $seed &

    sleep 20
    let "task=task+1"
  done
done