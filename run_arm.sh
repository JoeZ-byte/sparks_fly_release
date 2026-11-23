#!/usr/bin/env bash
# ARM 板端快速运行
set -e
cd "$(dirname "$0")/ARM"
python3 stream.py "$@"