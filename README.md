# CS436: Final Project
* By: Kaesi Manakkal

Training the models requires an NVIDIA GPU, as they train on CUDA.  Training the models takes a long time.
I have included the pre-trained models in the repo, and with the datasets in place, they can be tested without retraining.

## To train and test the mini-imagenet models: download reference from original readme
For mini-Imagenet experiments, please download [mini-Imagenet](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE) and put it in ./datas/mini-Imagenet and run proc_image.py to preprocess generate train/val/test datasets. (This process method is based on [maml](https://github.com/cbfinn/maml)).

Then you can train the models
```sh
$ cd miniimagenet
$ ./Train.sh
$ ./Test.sh
```

## To train and test the CIFAR-10 models:
The cifar-10 loader downloads the required dataset the first time you run it, so no need to do that.
Just train and test the models:
```sh
$ cd cifar10
$ ./Train.sh
$ ./Test.sh
```

Then view cifar10/logs/... and miniimagenet/logs/... to see results

## To generate confusion matrix for the miniimagenet models:
```sh
$ cd miniimagenet
$ python3 l2norm_confusion_mtx.py
$ python3 miniimagenet_confusion_mtx.py
$ python3 training_confusion_mtx.py

$ cd ../cifar-10
$ python3 training_confusion_mtx.py
```
The resultant images will be output to miniimagenet/images/
For the CIFAR-10 models, the procedure is similar, but I combined the confusion matrix and test scripts, so running ./Test.sh produces the confusion matricies.

NOTE:
Although the omniglot train and test scripts are in the repo, they have not yet been fully updated for python3 or pytorch > 0.4, so they won't run.

## Original Paper:
```
@inproceedings{sung2018learning,
  title={Learning to Compare: Relation Network for Few-Shot Learning},
  author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```