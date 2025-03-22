python main.py --data_dir '/home/ubuntu/xwy/dataset/MVSA_Single' \
--train_data_dir '/home/ubuntu/xwy/dataset/MVSA_Single' \
--test_data_dir '/home/ubuntu/xwy/dataset/MVSA_Single' \
--gpu 0 \
--save_dir './saved_models-single_ours/n600/complete_0.5' \
--lr 1e-4 \
--batch_size 2 \
--num_labels 600 \
--threshold 0.95 \
--num_train_iter 512 
# --optim AdamW