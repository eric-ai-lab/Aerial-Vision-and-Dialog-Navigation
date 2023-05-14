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

gdown 1shjHEH1xfB9h5ErcGFED1uIvKGWk88U5 -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/annotations/train_data.json

gdown 14vAVWpF8fp_L5J_0oUKlk1PwuQjUz3f2 -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/annotations/val_seen_data.json

gdown 1-_Si_v8BiI3m8n2GrBrS4EqYUt0Jpbci -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/annotations/val_unseen_data.json

gdown 1W4U4xqQo1_4_5x960FmhAi7b-C2dVyJ- -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/annotations/test_unseen_data.json
```

**Download pre-trained xview-yolov3 weights and model configuration**


```
gdown 1l_etD0Vm3-_hj7WTnfcIuDzgFVeMVQMr -O Aerial-Vision-and-Dialog-Navigation/datasets/XVIEW/pretrain_weights/best.pt
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
