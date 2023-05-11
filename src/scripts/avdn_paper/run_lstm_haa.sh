ngpus=1
seed=0

flag="--root_dir ../datasets

      --world_size ${ngpus}
      --seed ${seed}
      

      --feedback student

      --max_action_len 10
      --max_instr_len 100

      --lr 1e-5
      --iters 200000
      --log_every 2
      --batch_size 4
      --optim adamW

      --ml_weight 0.2      

      --feat_dropout 0.4
      --dropout 0.5
      
      --nss_w 0
      --nss_r 0

      --darknet_model_file ../datasets/XVIEW/pretrain_weights/yolo_v3.cfg
      --darknet_weight_file ../datasets/XVIEW/pretrain_weights/best.pt
      --eval_first True
      "



# train
# CUDA_VISIBLE_DEVICES='4' python xview_lstm/main.py --output_dir ../datasets/XVIEW/lstm_v8 $flag \

# eval
CUDA_VISIBLE_DEVICES='4' python xview_lstm/main.py --output_dir ../datasets/XVIEW/lstm_output $flag \
--resume_file ../datasets/XVIEW/lstm_haa/ckpts/best_val_unseen \
--test True
