#!/bin/sh
# shell to run Train.py with config file

# if the os is Mac, maybe exists .DS_Store file result in error while training
if [ "$(uname)" == "Darwin" ];then
        find . -name '*.DS_Store' -type f -delete
fi
python Train.py --config ./Config/Config.yaml