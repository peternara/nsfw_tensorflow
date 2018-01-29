# NSFW image detect based on deep neural networks Tensorflow models
This code is a demo to use inception network to detect NSFW images.

## Table of contents

<a href='#Data'>Preparing the dataset</a><br>
<a href='#Pretrained'>The pretrained model</a><br>
<a href='#Finetune'>Finetune the pretrained model</a><br>
<a href='#Export'>Exporting the Inference Graph</a><br>
<a href='#Freeze'>Freezing the exported Graph</a><br>
<a href='#Deploy'>Deploy the model</a><br>

<a id='Data'></a>
## Preparing the dataset
The dataset contains 260k images with 130k NSFW images as positive and 130k SFW images as negative. I don't own this dataset, [contact me](https://github.com/ZixuanLiang) for more information. 

The raw data is converted to TFRecord format. 

```shell
$ DATA_DIR=/tmp/data/nsfw
$ python convert_nsfw.py --dataset_dir="${DATA_DIR}"
```

<a id='Pretrained'></a>
## The pretrained model
The pretrained model we use to fine tune is [Inception V3](https://arxiv.org/abs/1512.00567). TF-slim file: [Code](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py). Checkpoint: [inception_v3_2016_08_28.tar.gz](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz). 

<a id='Finetune'></a>
## Finetune the pretrained model

First finetune the new layers for 100000 steps.

```shell
$ DATA_DIR=/tmp/data/nsfw
$ TRAIN_DIR=/tmp/nsfw-model
$ PRETRAINED_CHECKPOINT_DIR=/tmp/checkpoints
$ python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

Run evaluation.

```shell
$ DATA_DIR=/tmp/data/nsfw
$ TRAIN_DIR=/tmp/nsfw-model
$ python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_split_name=validation \
  --dataset_dir=${DATA_DIR} \
  --model_name=inception_v3

```

Fine-tune all the layers for 100000 steps.

```shell
$ python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_split_name=train \
  --dataset_dir=${DATA_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

Run evaluation.

```shell
$ DATA_DIR=/tmp/data/nsfw
$ TRAIN_DIR=/tmp/nsfw-model
$ python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR} \
  --dataset_split_name=validation \
  --dataset_dir=${DATA_DIR} \
  --model_name=inception_v3

```

<a id='Export'></a>
## Exporting the Inference Graph
Saves out a GraphDef containing the architecture of the model.

```shell
$ python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=/tmp/inception_v3_inf_graph.pb
```

<a id='Freeze'></a>
## Freezing the exported Graph
```shell
bazel build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=/tmp/inception_v3_inf_graph.pb \
  --input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
  --input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
  --output_node_names=InceptionV3/Predictions/Reshape_1
```
<a id='Deploy'></a>
## Deploy the model
I deploy the model on Heroku. Try the demo [here](https://zixuanliang.github.io/nsfw_demo.html).
