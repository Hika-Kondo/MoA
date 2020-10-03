#!/bin/bash

python /gbdt/src/main.py preprocess.function_name=select_features
python /gbdt/src/main.py preprocess.function_name=select_features_gen_cell_concat_pca_drop_low_variace
