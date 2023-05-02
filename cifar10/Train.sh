#!/bin/bash
LOGDIR="logs"
EPOCHS="10000"

mkdir -p ${LOGDIR}

# train 5way 5 shot
(time python3 cifar_train_fewshot.py -s 5 -b 15 -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_log_5way_5shot_${EPOCHS}ep.txt

# train 5way 1 shot
(time python3 cifar_train_fewshot.py -s 1 -b 15 -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_log_5way_1shot_${EPOCHS}ep.txt
