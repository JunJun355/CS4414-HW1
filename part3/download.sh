#!/bin/bash

DATA_DIR=./data
mkdir -p "${DATA_DIR}"

wget -O "${DATA_DIR}/queries_emb.json" https://cornell.box.com/shared/static/5xpdum6tuqx573kjrtcv3w1rbxdmqz4s.json
wget -O "${DATA_DIR}/passages1.json" https://cornell.box.com/shared/static/uoqqnwunfmz9w0c1akzzjqjwoecex3aw.json
wget -O "${DATA_DIR}/passages2.json" https://cornell.box.com/shared/static/2mp27kan81oa39t5djseyrd5w9wpdi0d.json