#!/bin/bash

#FEATURE=wmiftah
#FEATURE_CFG=wavelet/wmiftah/wmiftah.cfg
FEATURE=wtio
FEATURE_CFG=wavelet/wtio/wtio.cfg
#FEATURE=wdil
#FEATURE_CFG=wavelet/wdil/wtio.cfg

#DATASET=acacia-crassicarpa
#DATASET=eomf
DATASET=color

DATA_DIR=../dataset/arara-classify
OUT_DIR=../xprmt/feature

python extract.py $FEATURE $FEATURE_CFG $DATASET $DATA_DIR $OUT_DIR
