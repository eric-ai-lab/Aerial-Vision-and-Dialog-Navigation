# AVDN
Aerial Vision-and-Dialog Navigation

Todos:
- [x] Data released
- [x] Train code uploaded
- [x] Inference code uploaded and checkpoint released
- [ ] Eval.ai challenge setup

git pull https://github.com/UeFan/AVDN.git

# Download Data and Pre-trained Weights

Download xView data from https://challenge.xviewdataset.org/data-download.
Store the images from xView dataset in AVDN/datasets/XVIEW/train_images/:
`mkdir -p AVDN/datasets/XVIEW/train_images`

`cp -r XVIEW_images/*.tif AVDN/datasets/XVIEW/train_images/`

Download pre-trained xview-yolov3 weights and model configuration:


`gdown 1l_etD0Vm3-_hj7WTnfcIuDzgFVeMVQMr -O AVDN/datasets/XVIEW/pretrain_weights/best.pt`

# Training

`pip install -r requirements.txt`


Download AVDN datasets (https://sites.google.com/view/aerial-vision-and-dialog/home):

```
mkdir -p AVDN/datasets/XVIEW/annotations

gdown 1MNzgOH93C4ltymcY-jBe9RMwMKrNNIlD -O AVDN/datasets/XVIEW/annotations/train_data.json

gdown 14prBCIIMcfgnBnmeCz1SV2skDEyshCtr -O AVDN/datasets/XVIEW/annotations/val_seen_data.json

gdown 1CxNA8iSStvfMgw3hxtjtPvr2mEfbfFir -O AVDN/datasets/XVIEW/annotations/val_unseen_data.json
```

Run training:

`cd AVDN/src`

`bash scripts/train/run_et_haa.sh`