export GUROBI_HOME=/home/xxx/gurobi1103/linux64
export CUDA_VISIBLE_DEVICES=""
exp=(
    ocs_heuristic
    ocs_itv
    ocs_ilp
    ocs_ilp_itv
    ocs_helios
)

betas=(
    2100
)

export PYTHONPATH=/mnt/xxx/rapidnetsim-moe

for beta in ${betas[@]};
do
    mkdir beta_$beta
    cp -r base_conf/* beta_$beta
    cd beta_$beta
    for i in ${exp[@]};
    do
        cd $i
        python generate_ini_conf.py $beta &
        cd ..
    done
    cd ..
done

wait
unset PYTHONPATH
for beta in ${betas[@]};
do
    cd beta_$beta
    for i in ${exp[@]};
    do
        cd $i
        nohup rapidnetsim exp.ini > nohup.log 2>&1 &
        cd ..
    done
    cd ..
done
