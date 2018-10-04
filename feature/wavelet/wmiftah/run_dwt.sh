#!/bin/bash

EXE=dwt.py
OUT_DIR=../../xprmt/dwt
IN_DIR=../../dataset/bercak-resized

kernel = db2

python $EXE $kernel $IN_DIR $OUT_DIR
