# AVDN
Aerial Vision-and-Dialog Navigation

Todos:
- [x] Data released
- [x] Train code uploaded
- [x] Inference code uploaded and checkpoint released
- [ ] Eval.ai challenge setup

git pull https://github.com/UeFan/AVDN.git

# Download Data and Pre-trained Weights

**Download xView data** 

Our AVDN dataset uses satellite images from the xView dataset. Follow the instruction at https://challenge.xviewdataset.org/data-download to download xView dataset. 

Then move the images in xView dataset to under AVDN directory. (Assume the xView images are at ./XVIEW_images):
```
mkdir -p Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/train_images

cp -r XVIEW_images/*.tif Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/train_images/
```


**Download AVDN datasets**

 (https://sites.google.com/view/aerial-vision-and-dialog/home):

```
mkdir -p AVDN/datasets/XVIEW/annotations

gdown 1bdX5E1uEQXg8T8b5T7sh6WXb4V5rcG_Q -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/annotations/train_data.json

gdown 1y0PAW3bN1KtWPx9SkKuJU7o0c0tJsnzY -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/annotations/val_seen_data.json

gdown 1MYVAPeeS7Ydw4P7j1glvmYoE5yrN8Eo8 -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/annotations/val_unseen_data.json

gdown 14BijI07ukKCSDh3T_RmUG83z6Oa75M-U -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/annotations/test_unseen_data.json
```

**Download pre-trained xview-yolov3 weights and model configuration**


```
gdown 1Ke-pA5jpq1-fsEwAch_iRCtJHx6rQc-Z -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/pretrain_weights/best.pt

gdown 1n6RMWcHAbS6DA7BBug6n5dyN6NPjiPjh -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/pretrain_weights/yolo_v3.cfg
```

**Download the training checkpoints corresponding to the experiments in the AVDN paper**


```
mkdir -p Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/et_haa/ckpts/

mkdir -p Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/lstm_haa/ckpts/

gdown 1fA6ckLVA-gsiOmWmOMkqJggTLbiJpFBI -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/et_haa/ckpts/best_val_unseen

gdown 1RYjo_vc5m5ZRUcjIFojZjke8RhlfX90I -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/lstm_haa/ckpts/best_val_unseen
```

# Training and Evaluation
**Install requirements**

```
pip install -r requirements.txt
```



**Run training or evaluation:**

The script, `scripts/avdn_paper/run_et_haa.sh`, includes commands for train and evaluate Human Attention Aided Transformer (HAA-Transformer) model.

The script, `scripts/avdn_paper/run_lstm_haa.sh`, includes commands for train and evaluate Human Attention Aided LSTM (HAA-LSTM) model.



```
cd Aerial-Vision-and-Dialog-Navigation/src

# For Human Attention Aided Transformer model
bash scripts/avdn_paper/run_et_haa.sh 

# For Human Attention Aided LSTM model
bash scripts/avdn_paper/run_lstm_haa.sh 
```
