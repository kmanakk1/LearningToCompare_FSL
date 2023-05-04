#!/bin/bash
LOGDIR="logs"
EPOCHS="10000"

mkdir -p ${LOGDIR}

# train baseline -  5way 5shot
(time python3 cifar_train_fewshot.py -s 5 -b 15 -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_log_10way_5shot_${EPOCHS}ep.txt

# train with l2 and dropout - 5way 5shot
(time python3 l2norm_cifar_train_fewshot.py -s 5 -b 15 -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_l2norm_log_10way_5shot_${EPOCHS}ep.txt
