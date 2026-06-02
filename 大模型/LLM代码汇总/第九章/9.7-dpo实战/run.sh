#!/bin/bash

nohup accelerate launch --config_file accelerate_config.yaml mini_qwen_dpo.py > output_dpo.log 2>&1 &


