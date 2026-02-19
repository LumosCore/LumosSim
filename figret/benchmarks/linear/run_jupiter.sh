export PYTHONPATH='xxx/rapidAIsim_ai4topo'
for i in $(seq $1 $2)
do
    nohup python window_algorithm_run.py --topo_name LumosCore-256spine-2048gpu --TE_solver Jupiter --test_hist_names $i --hist_len 3 > jupiter$i.log 2>&1 &
done
