#!/bin/bash

#experiment parameters
#apps="lstm resnet20 resnet50"
apps="resnet50"
nodes="2"
#topologies="complete ring torus"
topologies="complete"

#summit parameters
gpu_per_node=6

#script for experiment runs
mkdir -p logs
app_run(){
    local app=$1
    local script=original_tplt/run-${app}.sh.tplt
    local lsf=original_tplt/job.lsf
    per(){
        local node=$1
        local topology=$2 
        local rank_per_node=$3  

        NRANK=$(($node*$gpu_per_node))

        local time=`date +%s`
        local job=job_${app}_${node}_${topology}_${rank_per_node}_${time}.lsf
        local run=run_${app}_${node}_${topology}_${rank_per_node}_${time}.sh

        #specify submission file
        cp $lsf $job 
        sed -i "s/APP/${app}/g" $job
        sed -i "s/NODE/${node}/g" $job
        sed -i "s/TOPOLOGY/${topology}/g" $job
        sed -i "s/TIMESTAMP/${time}/g" $job
        sed -i "s/NRANK/${NRANK}/g" $job
        sed -i "s/PER/${rank_per_node}/g" $job
         

        #specify run script file
        cp $script $run 
        if [[ $rank_per_node == "6" ]]; then 
            world=$(python -c "print(','.join(['0,1,2,3,4,5']*$node))")
        elif [[ $rank_per_node == "1" ]]; then
            world=$(python -c "print(','.join(['0']*$NRANK))")
        fi

        sed -i "s/APP/$app/g" $run 
        sed -i "s/NRANK/$NRANK/g" $run 
        sed -i "s/GPURANK/$world/g" $run 
        sed -i "s/TOPOLOGY/$topology/g" $run
        sed -i "s/NODE/$node/g" $run
        sed -i "s/TIMESTAMP/$time/g" $run
        sed -i "s/PER/$rank_per_node/g" $run
        sed -i "s/NSUB/$rank_per_node/g" $run


        bsub $job
    }
    for node in $nodes; do
        for topology in $topologies; do
            for rank_per_node in "1"; do
                per $node $topology $rank_per_node 
            done
        done
    done
}
for app in $apps; do
   app_run $app
done 
