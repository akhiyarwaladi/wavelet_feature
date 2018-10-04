#!/bin/bash

EXE=flbp.py
OUT_DIR=../../xprmt/flbp
IN_DIR=../../dataset/bercak-resized

N_NEIGHBORS=8
RADIUS=2

python $EXE $N_NEIGHBORS $RADIUS $IN_DIR $OUT_DIR
