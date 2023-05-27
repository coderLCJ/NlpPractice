#!/bin/sh

cd .. 
current_dir=$(pwd) 
export PYTHONPATH=$PYTHONPATH:$current_dir/run:$current_dir/idl:$current_dir

python3 ./server/server.py