#!/bin/bash
# kmanakk1 - script to repeat training for models demonstrated in the paper for miniimagenet
LOGDIR="logs"
mkdir -p ${LOGDIR}

# test baseline
(time python3 cifar_confusion_mtx.py -s 5)  2>&1 | tee ${LOGDIR}/test_10way5shot.txt

# test with l2 regularization and dropout
(time python3 l2norm_confusion_mtx.py -s 5)  2>&1 | tee ${LOGDIR}/test_l2norm_10way5shot.txt
