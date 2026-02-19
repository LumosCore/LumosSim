export PYTHONPATH='/mnt/d/Code/PyCharm/rapidAIsim-ai4topo'
nohup python figret.py --topo_name LumosCore-256spine-2048gpu --mode train --num_layer 3 --epochs 30 --batch_size 16 --beta $1 --hist_len 3 --train_hist_names 48-49 --test_hist_names 50 --lr 0.001 --single_hist_size 2000 --dataset_label 2200_500_$1 > train_$1.log 2>&1 &
