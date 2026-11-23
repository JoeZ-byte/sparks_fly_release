#!/usr/bin/env bash
# x86 快速运行
set -e
cd "$(dirname "$0")/X86"
python3 stream.py "$@"