# FYP

## Install Dependencies:

```
$ pip install -r requirements.txt
```

## Converting VOC to COCO format for annotations

### 1. Make labels.txt

labels.txt if need for making dictionary for converting label to id.

**Sample labels.txt**

```txt
Label1
Label2
...
```

In order to get all labels from your `*.xml` files, you can use this command in shell:

```
grep -REoh '<name>.*</name>' /Path_to_Folder | sort | uniq
```

This will search for all name tags in `VOC.xml` files, then show unique ones. You can also go further and create `labels.txt` file.

```
grep -ERoh '<name>(.*)</name>' /Path_to_folder | sort | uniq | sed 's/<name>//g' | sed 's/<\/name>//g' > labels.txt
```

### 2. Split your data into train,test and validation set. (Annotations + Images)

### 3. Run script for each dataset

##### 3.1 Usage 1(Use ids list)

```bash
$ python voc2coco.py \
    --ann_dir /path/to/annotation/dir \
    --ann_ids /path/to/annotations/ids/list.txt \
    --labels /path/to/labels.txt \
    --output /path/to/output.json \
    <option> --ext xml
```

##### 3.2 Usage 2(Use annotation paths list)

**Sample paths.txt**

```txt
/path/to/annotation/file.xml
/path/to/annotation/file2.xml
...
```

## To run training:

### 1. Configure [train.yaml](./config/train.yaml) based on the configurations intended.

Some important variables in the yaml file are included below. <br>
Output is saved under 'experiment', where <br>
'name': name_of_experiment <br>
'folder': The folder where the output of the training will be saved.

'phase': train/detect <br>

### Under 'train': <br>

'ann_path': path of annotations for training e.g. /Users/grego/Downloads/FYP/data/train_output.json<br>
'img_path': path of images for training e.g. /Users/grego/Downloads/FYP/data/train_images <br>
'img_size': size of image <br>
'transform': strong/weak based on the type of augmentation you want. Strong for more drastic augmentations. <br>
^variables above are the same for 'val' <br>

### Under Model:

'load_weights': True/False <br>
'weights_path': path to weights file e.g. /Users/grego/Downloads/FYP/best.h5 <br>
'pretrain_weights': path to pretrained weights file /Users/grego/Downloads/FYP/mobilenet_v2.pth.tar <br>
'num_classes': number of classes for detection <br>
'img_size': size of image model accepts <br>
'feature_maps': number of feature maps <br>
'name': name of model e.g. retina_net <br>
'backbone': name of backbone used e.g. mobilenet_v2, shufflenet_X1,shufflenet_X0_75_V4(optimal backbone)<br>

'batch_size': size of batch e.g. 16 <br>
'num_epochs': number of epochs e.g. 50 <br>
'early_stopping': number of epochs for early stopping e.g. 15 <br>

### 2. Train the model:

```
$ python train.py
```

## To run evaluation(inference):

### 1. Configure [test.yaml](./config/test.yaml) based on the configurations intended. Important to make sure the variables listed below is correct.

'weights_path': path of weights saved from training. e.g. /Users/grego/Downloads/FYP/tiger_detection_shufflenet_X0_75_V4/last.h5 <br>
'backbone': name of backbone used e.g. shufflenet_X0_75_V4 <br>
'filepath' : path of weights file after training /Users/grego/Downloads/FYP/tiger_detection_shufflenet_X0_75_V4/last.h5 <br>

### 2. Evaluate the model:

```
$ python eval.py
```
