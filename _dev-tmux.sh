#!/bin/bash

output_dir='./output'
conda_env_name="jdt"

tmux new-session \; \
    send-keys 'ngrok start --all' C-m \; \
    split-window -v \; \
    send-keys 'conda activate '$conda_env_name'; jupyter notebook --no-browser --port=7111' C-m \; \
    split-window -v \; \
    send-keys 'conda activate '$conda_env_name'; python -m visdom.server' C-m \; \
    split-window -v \; \
    send-keys 'conda activate '$conda_env_name'; tensorboard --logdir='$output_dir'/tb_logs' C-m \; \