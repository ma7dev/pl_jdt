#!/bin/bash
conda_env_name="jdt"
tmux new-session \; \
    send-keys 'conda activate '$conda_env_name C-m \; \
    split-window -h \; \
    send-keys 'watch -n1 "nvidia-smi --query-gpu=memory.used --format=csv"' C-m \; \