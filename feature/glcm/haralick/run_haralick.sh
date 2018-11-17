#!/bin/bash

EXE=haralick.py
OUT_DIR=../../../xprmt/glcm
IN_DIR=../../../dataset/arara-classify/DataPH2

N_NEIGHBORS=8
RADIUS=2

python $EXE $IN_DIR $OUT_DIR