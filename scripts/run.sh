#!/bin/bash
# 가상환경을 켜고 파이썬 스크립트를 실행한다. 진행 막대 줄은 걸러서 로그를 짧게.
# 사용법: bash scripts/run.sh scripts/02_make_stimuli_and_record.py
cd "$(dirname "$0")/.."
source ~/.local/bin/env
source .venv/bin/activate
export PYTHONUNBUFFERED=1
python "$@" 2>&1 | grep --line-buffered -v -E "^\s*$|Batch:|it/s\]|s/it\]"
exit ${PIPESTATUS[0]}
