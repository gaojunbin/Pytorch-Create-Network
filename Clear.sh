#!/bin/sh
log_dir=./logs
checkpoint_dir=./checkpoint
if [ "`ls -A ${log_dir}`" == "" ];then 
        echo "暂无日志"
else
        rm -f -r ${log_dir}/*
        rm -f -r ${log_dir}/.*
        echo "日志文件已清空"
fi
if [ "`ls -A ${checkpoint_dir}`" == "" ];then 
        echo "暂无模型"
else
        rm -f -r ${checkpoint_dir}/*
        rm -f -r ${checkpoint_dir}/.*
        echo "模型已清空"
fi
find . -name '__pycache__' -type d -exec rm -rf {} \;