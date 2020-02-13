#! /bin/bash

set -e 

if [ "$#" -ne 2 ];then
    echo "Usage run.sh [svgd|map_kfac|svgd_kfac|mixture_kfac|sgld|psgld] lr"
    exit 1
fi

method=$1
learning_rate=$2


declare -A datasets=( [year]=10)
declare -A batchsize=( [year]=1000 )


for ds in "${!datasets[@]}"
do
    for ((i=1; i<=20; i++ ))
    do
        python trainer.py --method ${method} --dataset $ds --trial $i --n_epoches ${datasets[$ds]}  --learning_rate ${learning_rate} --batch_size ${batchsize[$ds]}
    done
done


