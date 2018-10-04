#!/bin/bash

EXE=dwt.py
OUT_DIR=../../xprmt/dwt
IN_DIR=../../eomf

kernel = db2

python $EXE $kernel $IN_DIR $OUT_DIR
