export PYTHONPATH='/mnt/d/Code/PyCharm/rapidAIsim-ai4topo/'
# $3 is 'train' or 'test'
for i in $(seq $1 $2)
do
    nohup python generate_dataset.py --topo_name LumosCore-256spine-2048gpu --beta 0.1 --${3}_hist_names $i > nohup${i}_${3}.out 2>&1 &
done
# --beta 1
