#!/bin/bash
# kmanakk1 - script to repeat training for models demonstrated in the paper for miniimagenet
LOGDIR="logs"
mkdir -p ${LOGDIR}

# test baseline
(time python3 miniimagenet_test_few_shot.py -w 5 -s 5)  2>&1 | tee ${LOGDIR}/test_5way5shot.txt

# test with l2 regularization and dropout
(time python3 l2norm_test_fewshot.py -w 5 -s 5)  2>&1 | tee ${LOGDIR}/test_l2norm_5way5shot.txt