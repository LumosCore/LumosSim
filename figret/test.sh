export PYTHONPATH='/mnt/d/Code/PyCharm/rapidAIsim-ai4topo'
nohup python figret.py --topo_name LumosCore-256spine-2048gpu --mode test --num_layer 3 --epochs 10 --batch_size 16 --alpha 0.01 --hist_len 3 --test_hist_names 50 --lr 0.001 --single_hist_size 2000 --dataset_label 2200_500_0.1 > test_1.log 2>&1 &
