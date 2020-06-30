#!/bin/sh

if [ "$(uname)" == "Darwin" ];then
        find . -name '*.DS_Store' -type f -delete
fi
python Train.py --config ./Config/Config.yaml