#!/bin/bash
# kmanakk1 - script to repeat training for models demonstrated in the paper for omniglot
LOGDIR="logs"
EPOCHS="100000"

mkdir -p ${LOGDIR}

trainModel() {
    W=$1
    S=$2
    B=$3

    # use oneshot trainer
    [ "$S" == 1 ] && (time python3 omniglot_train_one_shot.py -w ${W} -s ${S} -b ${B} -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_log_${W}way_${S}shot_${EPOCHS}ep.txt

    # use fewshot trainer
    [ "$S" == 1 ] || (time python3 omniglot_train_few_shot.py -w ${W} -s ${S} -b ${B} -e ${EPOCHS}) 2>&1 | tee ${LOGDIR}/train_log_${W}way_${S}shot_${EPOCHS}ep.txt
}

testModel() {
    W=$1
    S=$2
    [ "$S" == 1 ] && (time python3 omniglot_test_one_shot.py -w ${W} -s ${S})  2>&1 | tee ${LOGDIR}/test_${W}way${S}shot.txt
    [ "$S" == 1 ] || (time python3 omniglot_test_few_shot.py -w ${W} -s ${S})  2>&1 | tee ${LOGDIR}/test_${W}way${S}shot.txt
}

# Train Models
trainModel 5  1 19        # 5way  1shot
trainModel 20 1 10        # 20way 1shot

trainModel 5  5 15        # 5way  5shot
trainModel 20 5 15        # 20way 5shot

# Test Models
testModel 5 1           # 5way 1shot
testModel 20 1          # 20way 1shot

testModel 5 5           # 5way 5shot
testModel 20 5          # 20way 5shot
