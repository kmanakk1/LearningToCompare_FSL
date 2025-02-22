#!/bin/bash
# kmanakk1 - script to repeat training for models demonstrated in the paper for miniimagenet
LOGDIR="logs"
EPOCHS="100000"

mkdir -p ${LOGDIR}

# train 5way 1 shot
#(time python3 miniimagenet_train_one_shot.py -w 5 -s 1 -b 15 -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_log_5way_1shot_${EPOCHS}ep.txt
# test 5way1shot
#(time python3 miniimagenet_test_one_shot.py -w 5 -s 1)  2>&1 | tee ${LOGDIR}/test_5way1shot.txt


## BASELINE
# train 5way 5 shot
(time python3 miniimagenet_train_few_shot.py -w 5 -s 5 -b 10 -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_log_5way_5shot_${EPOCHS}ep.txt

## WITH L2 regularization and Dropout
(time python3 l2norm_train_fewshot.py -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_l2norm_log_5way_5shot_${EPOCHS}ep.txt
