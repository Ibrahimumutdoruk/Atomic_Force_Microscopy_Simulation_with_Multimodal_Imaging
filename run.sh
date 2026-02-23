#!/bin/bash
set -e

export MPLBACKEND=TkAgg

python3 -m pip install --user -r requirements.txt
python3 AFM_simulation.py