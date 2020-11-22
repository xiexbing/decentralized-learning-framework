import os
import json
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
from scipy import stats
import subprocess as sp
from datetime import datetime
from matplotlib import cm
import math

experiments = [["wikitext2/rnn_lm/test", 300], ["cifar10/resnet20/test",300],[ "imagenet/resnet50/test", 90]]

result = "result"
jason_dir = "/gpfs/alpine/stf008/scratch/bing/dl/federated-learning/dl_code/checkpoint"
gpu_per_node = 6

def read_file(data_file):
    d_f = open(data_file, 'r')
    dfile = d_f.readlines()
    d_f.close()

    return dfile


def process_run(record_file):

    train_raw = []
    test_raw = []
    test_local_model = []
    test_local_avg = []
    test_local_full = []
    test_avg_full = []

    with open(record_file, "r") as read_file:
        data = json.load(read_file)
   
    #split the results to train and test
    for line in data:
        for per in line:
            if 'train' == line[per]:
                train_raw.append(line)
            elif 'test' == line[per]:
                test_raw.append(line)
   
    #process test, 4 types of test results 
    for line in test_raw:
        for per in line:
            if 'local_model_avg' == line[per]:
                test_local_avg.append(line)
            elif 'local_model' == line[per]:
                test_local_model.append(line)                
            elif 'eval_local_model_on_full_training_data' == line[per]:
                test_local_full.append(line)
            elif 'eval_averaged_model_on_full_training_data' == line[per]:
                test_avg_full.append(line) 


    return [train_raw, test_local_model, test_local_avg, test_local_full, test_avg_full] 


def analyze_test(test_local_model, test_local_avg, test_local_full, test_avg_full):

    if len(test_local_full) < 300 or len(test_avg_full) < 300:
        return 1

    sorted_local_model = sorted(test_local_model, key = lambda i: i['epoch'])
    sorted_local_avg = sorted(test_local_avg, key = lambda i: i['epoch'])
    print (sorted_local_model[:3])
 
    top_local = []
    top_avg = []
    top_local_full = []
    top_avg_full = []

    for per in sorted_local_model:
        tops = [per['top1'], per['top5']]  
        top_local.append(tops)
 
    for per in local_full:
        tops = [per['top1'], per['top5']]  
        top_local_full.append(tops)

    for per in avg_full:
        tops = [per['top1'], per['top5']]  
        top_avg_full.append(tops)

    for per in sorted_avg:
        top_avg.append(per['best_perf'])

    return [top_local, top_avg, top_local_full, top_avg_full] 



def analyze_train(train_raw, run_dir, train_epoch):

    sorted_train = sorted(train_raw, key = lambda i: i['epoch'])
 
    train_times = []
    for i in range(train_epoch):
        epoch_train = 0
        comm_train = 0
        compute_train = 0
        data_train = 0
        size_train = 0
        for per in sorted_train:
            #track time
            if 'complete' in run_dir:
                epoch_records = ['backward_pass', 'forward_pass', 'load_data', 'sync.apply_grad', 'sync.get_data', 'sync.sync', 'sync.unflatten_grad', 'sync_complete']  
                data_records = ['load_data', 'sync.apply_grad', 'sync.get_data', 'sync.sync', 'sync.unflatten_grad', 'sync_complete']  
            else: 
                epoch_records = ['backward_pass', 'forward_pass', 'load_data', 'sync.apply_grad', 'sync.get_data', 'sync.sync', 'sync.update_model', 'sync_complete']    
                data_records = ['load_data', 'sync.apply_grad', 'sync.get_data', 'sync.sync', 'sync.update_model', 'sync_complete']    


            #track data size
            size_records = ['n_bits_to_transmit']
            comm_records = ['sync.sync']
            compute_records = ['backward_pass', 'forward_pass']
 
            train_time = sum(per[k] for k in epoch_records)
            comm_time = sum(per[k] for k in comm_records)
            compute_time = sum(per[k] for k in compute_records)
            data_time = sum(per[k] for k in data_records)
            data_size = sum(per[k] for k in size_records)
 
            if per['epoch'] <= i+1 and per['epoch'] > i:
                epoch_train += train_time
                comm_train += comm_time
                data_train += data_time
                compute_train += compute_time
                size_train += data_size
#        print (epoch_train, comm_train, local_train)
        train_times.append([epoch_train, comm_train, compute_train, data_train, size_train/8])

    return train_times        

def plot_train(train_data, run_name, train_epoch):

    rdir = os.path.join(result, run_name)
    if not os.path.isdir(rdir):
        os.makedirs(rdir)

    train_data.sort(key=lambda x:x[0])

    #analyze train times across epoch  
    train_sum = []
    comm_sum = []
    compute_sum = []
    data_sum = []
    size_sum = []
    network_percent = []

    for rank_sum in train_data:
        train_rank = []
        comm_rank = []
        size_rank = []
        compute_rank = []
        data_rank = []
        network_percent_rank = []

        for epoch in rank_sum[1]: 
            [train_time, comm_time, compute_time, data_time, data_size] = epoch
            print (train_time, comm_time, compute_time, data_time)
            compute_rank.append(compute_time) 
            data_rank.append(data_time) 
            train_rank.append(train_time)
            comm_rank.append(comm_time)
            size_rank.append(data_size)
            network_percent_rank.append(100*comm_time/train_time)

        data_sum.append(data_rank)
        compute_sum.append(compute_rank)
        train_sum.append(train_rank)
        comm_sum.append(comm_rank)
        size_sum.append(size_rank)
        network_percent.append(network_percent_rank)


    #plot train time per epoch

    tmax = 0
    fig, ax = plt.subplots(4)
    x = range(train_epoch)
    for rank in train_sum:
        ax[0].plot(x, rank, linewidth =1)
        if np.max(rank) > tmax:
            tmax = np.max(rank)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Train")
    ax[0].set_ylim(0, tmax)


    #compute time
    for rank in compute_sum:
        ax[1].plot(x, rank, linewidth =1)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Compute")
    ax[1].set_ylim(0, tmax)

    #data time
    for rank in data_sum:
        ax[2].plot(x, rank, linewidth =1)
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Data")
    ax[2].set_ylim(0, tmax)


    #sync time
    for rank in comm_sum:
        ax[3].plot(x, rank, linewidth =1)
    ax[3].set_xlabel("Epoch")
    ax[3].set_ylabel("Sync")
    ax[3].set_ylim(0, tmax)


    Name = os.path.join(rdir, "times.pdf")
    plt.savefig(Name)
    plt.close()


    plt.figure()
    for rank in size_sum:
        plt.plot(x, rank, linewidth =1)
    plt.xlabel("Epoch")
    plt.ylabel("Data Size of an Epoch, Unit: Byte")
    Name = os.path.join(rdir, "data_size.pdf")
    plt.savefig(Name)
    plt.close()

    plt.figure()
    for rank in network_percent:
        plt.plot(x, rank, linewidth =1)
    plt.xlabel("Epoch")
    plt.ylabel("Sync Percent for an Epoch")
    plt.ylim(0, 100)
    Name = os.path.join(rdir, "sync_percent.pdf")
    plt.savefig(Name)
    plt.close()



def plot_test(test_data, run_name):

    rdir = os.path.join(result, run_name)
    if not os.path.isdir(rdir):
        os.makedirs(rdir)

    test_data.sort(key=lambda x:x[0])

    #analyze train times across epoch  
    top_local_sum = []
    top_avg_sum = []
    top_local_full_1 = []
    top_local_full_5 = []
    top_avg_full_1= []
    top_avg_full_5 = []


#    top_local, top_avg, top_local_full, top_avg_full
    for rank_sum in test_data:
        [top_local, top_avg, top_local_full, top_avg_full] = rank_sum[1]
        top1_rank = []
        top5_rank = []

        for epoch in top_local: 
            [top1, top5] = epoch
            top1_rank.append(top1)
            top5_rank.append(top5)
        for epoch in top_local_full: 
            [top1, top5] = epoch
            top_local_full_1.append(top1)
            top_local_full_5.append(top5)

        for epoch in top_avg_full: 
            [top1, top5] = epoch
            top_avg_full_1.append(top1)
            top_avg_full_5.append(top5)

        top_local_sum.append([top1_rank, top5_rank])
        top_avg_sum.append(top_avg)


    x = range(train_epoch)
    plt.figure()
    for rank in top_local_sum:
        plt.plot(x, rank[0], linewidth =1)
    plt.xlabel("Epoch")
    plt.ylabel("Top 1, accuracy")
    plt.ylim(0, 100)
    Name = os.path.join(rdir, "local_top1.pdf")
    plt.savefig(Name)
    plt.close()

    plt.figure()
    for rank in top_local_sum:
        plt.plot(x, rank[1], linewidth =1)
    plt.xlabel("Epoch")
    plt.ylabel("Top 5, accuracy")
    plt.ylim(0, 100)
    Name = os.path.join(rdir, "local_top5.pdf")
    plt.savefig(Name)
    plt.close()


    x = range(len(top_local_full_1))
    plt.plot(x, top_local_full_1, linewidth =1, label = 'local full top 1')
    plt.plot(x, top_local_full_5, linewidth =1, label = 'local full top 5')
    plt.plot(x, top_avg_full_1, linewidth =1, label = 'avg full top 1')
    plt.plot(x, top_avg_full_5, linewidth =1, label = 'avg full top 5')
    plt.legend(fontsize=5, loc=1)
    plt.xlabel("Rank")
    plt.ylabel("Accuracy, percentile")
    plt.ylim(0, 100)
    Name = os.path.join(rdir, "full_train.pdf")
    plt.savefig(Name)
    plt.close()

 
def main():

    if os.path.isdir(result):
        shutil.rmtree(result)

    for experiment in experiments:
        [name, train_epoch] = experiment
        edir = os.path.join(jason_dir, name)
        for run in os.listdir(edir):
            rdir = os.path.join(edir, run)
            run_train = []
            run_test = []
            [dataName, nodes, topology, world, timestamp] = run.split('_') 
            ranks = int(nodes) * gpu_per_node 
            print (rdir)
            for rank in os.listdir(rdir):
                rank_dir = os.path.join(rdir, rank)
                if os.path.isdir(rank_dir):
                    train_raw = []
                    test_local_model = []
                    test_local_avg = []
                    test_local_full = []
                    test_avg_full = []
                    for record in os.listdir(rank_dir):
                        if '.json' in record:
                            record_dir = os.path.join(rank_dir, record)
                            [curr_train, curr_test_local_model, curr_test_local_avg, curr_test_local_full, curr_test_avg_full] = process_run(record_dir)
                            train_raw += curr_train
                            test_local_model += curr_test_local_model
                            test_local_avg += curr_test_local_avg
                            test_local_full += curr_test_local_full
                            test_avg_full += curr_test_avg_full
                   
                    train_time = analyze_train(train_raw, record_dir, train_epoch)
                    if len(train_time) != train_epoch:
                        print ("incomplete run", rdir)
                        break
                    else:
                        run_train.append([int(rank), train_time]) 

            if len(run_train) == ranks:
                plot_train(run_train, run, train_epoch)                        

main()                                   
