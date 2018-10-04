#!/bin/bash

EXE=wtio.py
OUT_DIR=../../xprmt/wpacket
IN_DIR=../../dataset/arara

KERNEL=db2
LEVEL=5
SUBBAND=ll
DOWNSAMPLING=1

python $EXE $KERNEL $LEVEL $SUBBAND $DOWNSAMPLING $IN_DIR $OUT_DIR