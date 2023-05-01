ngpus=1
seed=0

outdir=../datasets/XVIEW/et_v8_student_haa
flag="--root_dir ../datasets
      --output_dir ${outdir}
      
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
      
      --nss_w 0.1
      --nss_r 0

      --darknet_model_file ../datasets/XVIEW/pretrain_weights/yolo_v3.cfg
      --darknet_weight_file ../datasets/XVIEW/pretrain_weights/best.pt
      --eval_first True
      "



# train
CUDA_VISIBLE_DEVICES='5'  python xview_et/main.py $flag \
      # -train_val_on_full \
      # --resume_file ../datasets/XVIEW/et_v3_no_haa_flat/ckpts/best_val_unseen

      # -num_replacement\
      # --resume_file ../datasets/XVIEW/next61_no_sali/ckpts/best_val_unseen
      # 

