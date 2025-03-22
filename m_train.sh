python main.py --data_dir '/home/ubuntu/xwy/dataset/MVSA' \
--train_data_dir '/home/ubuntu/xwy/dataset/MVSA' \
--test_data_dir '/home/ubuntu/xwy/dataset/MVSA' \
--gpu 0 \
--save_dir './saved_models-multiple_ours/n600/msa' \
--lr 1e-4 \
--num_labels 600 \
--batch_size 2 \
--num_train_iter 256 \
--threshold 0.95
# --eval_batch_size 32
# --class_weight [0.2, 0.5, 0.3]
# --resume \
# --load_path 'saved_models-multiple/main/model_best.pth'