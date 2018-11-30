#!/bin/bash

# FEATURE=wmiftah
# FEATURE_CFG=wavelet/wmiftah/wmiftah.cfg
# FEATURE=wtio
# FEATURE_CFG=wavelet/wtio/wtio.cfg
#FEATURE=wdil
#FEATURE_CFG=wavelet/wdil/wtio.cfg
# FEATURE=slbp
# FEATURE_CFG=lbp/slbp/slbp.cfg
FEATURE=haralick
FEATURE_CFG=glcm/haralick/haralick.cfg

#DATASET=acacia-crassicarpa
#DATASET=melanoma_binary
#DATASET=melanoma_binary_augmented
DATASET=melanoma_hair
#DATASET=DataPH2_lesion_hairremove
#DATASET=DataPH2_lesion_hairremove_augmented
#DATASET=DataPH2_lesion
#DATASET=DataPH2_lesion_augmented

DATA_DIR=../dataset/arara-classify
OUT_DIR=../xprmt/feature

python extract.py $FEATURE $FEATURE_CFG $DATASET $DATA_DIR $OUT_DIR
