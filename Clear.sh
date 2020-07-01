#!/bin/sh
# clear the cache file and restart to train and inference
log_dir=./logs
checkpoint_dir=./checkpoint
if [ "`ls -A ${log_dir}`" == "" ];then 
        echo "no logs exist"
else
        rm -f -r ${log_dir}/*
        rm -f -r ${log_dir}/.*
        echo "logs has been deleted all"
fi
if [ "`ls -A ${checkpoint_dir}`" == "" ];then 
        echo "no checkpoints exist"
else
        rm -f -r ${checkpoint_dir}/*
        rm -f -r ${checkpoint_dir}/.*
        echo "checkpoints has been deleted all"
fi
find . -name '__pycache__' -type d -exec rm -rf {} \;