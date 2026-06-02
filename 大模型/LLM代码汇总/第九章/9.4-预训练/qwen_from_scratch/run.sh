#!/bin/bash
nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_pt.py > output_pt.log 2>&1 &
