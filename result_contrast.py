#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:00:16 2020

@author: marshallyin
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
from decimal import Decimal


# loss = ['ori','loss_logistic','loss_poisson']

# emb_dim = ['16','ori','64','128']

# n_iter = ['5','ori','20','40']

# bat = ['128','ori','512','1024']

# l2 = ['ori','02','04','08']

lr = ['ori','2e','4e','8e']


with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_ori.csv","r") as f:
    reader_ori = csv.reader(f)
    result_list = list(reader_ori)
    ori_vm_value = []
    ori_vm_name = []
    ori_time_name = []
    ori_sys_runtime = []
    ori_cpu_time = []
    ori_user_runtime = []
    ori_RMSE_value = []
    sample_name = []
    for i in range(2,14):
        ori_vm_value.append(int(result_list[1][i]))
        ori_vm_name.append(result_list[0][i])
    for i in range(14,16):
        ori_time_name.append(result_list[0][i])
    for j in range(1,11):
        ori_sys_runtime.append(float(result_list[j][14]))
        ori_user_runtime.append(float(result_list[j][15]))
        ori_cpu_time.append(float(result_list[j][16]))
        ori_RMSE_value.append(float(result_list[j][17]))
        sample_name.append(result_list[j][0][:-7])
    
with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_loss_logistic.csv","r") as f:
    reader_logistic = csv.reader(f)
    result_list_logistic = list(reader_logistic)
    logistic_vm_value = []
    logistic_sys_runtime = []
    logistic_user_runtime = []
    logistic_cpu_time = []
    logistic_RMSE_value = []   
    for i in range(2,14):
        logistic_vm_value.append(int(result_list_logistic[1][i]))
    for j in range(1,11):
        logistic_sys_runtime.append(float(result_list_logistic[j][14]))
        logistic_user_runtime.append(float(result_list_logistic[j][15]))
        logistic_cpu_time.append(float(result_list_logistic[j][16]))
        logistic_RMSE_value.append(float(result_list_logistic[j][17]))

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_loss_poisson.csv","r") as f:
    reader_poisson = csv.reader(f)
    result_list_poisson = list(reader_poisson)
    poisson_vm_value = []
    poisson_sys_runtime = []
    poisson_user_runtime = []
    poisson_cpu_time = []
    poisson_RMSE_value = []   
    for i in range(2,14):
        poisson_vm_value.append(int(result_list_poisson[1][i]))
    for j in range(1,11):
        poisson_sys_runtime.append(float(result_list_poisson[j][14]))
        poisson_user_runtime.append(float(result_list_poisson[j][15]))
        poisson_cpu_time.append(float(result_list_poisson[j][16]))
        poisson_RMSE_value.append(float(result_list_poisson[j][17]))

plt.figure(dpi=500,figsize=(18,6))
width = 0.3
index = np.arange(len(ori_vm_name))
plt.bar(index,ori_vm_value,width,color='Skyblue',label='loss:regression',tick_label = ori_vm_name) 
plt.bar(index - width,logistic_vm_value,width,color='Indianred',label='loss:logistic')
plt.bar(index + width,poisson_vm_value,width,color='orange',label='loss:poisson')
plt.legend(['loss:regression','loss:logistic','loss:poisson'])
plt.title("The difference of Virtual Memory with 3 models --- loss")
for a,b in zip(index,ori_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=6)
for a,b in zip(index - width,logistic_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=6)
for a,b in zip(index + width,poisson_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=6)
plt.ylabel("KB")   
plt.savefig('./figures_result_loss/loss_Vm_bar.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, ori_sys_runtime,c='Skyblue',label='loss:regression')
plt.plot(sample_name, logistic_sys_runtime,c='Indianred',label='loss:logistic')
plt.plot(sample_name, poisson_sys_runtime,c='orange',label='loss:poisson')
plt.legend(['loss:regression','loss:logistic','loss:poisson'])
for a,b in zip(sample_name,ori_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,logistic_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,poisson_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("System Running Time with 3 models --- loss")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_loss/loss_system_runtime_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, ori_user_runtime,c='Skyblue',label='loss:regression')
plt.plot(sample_name, logistic_user_runtime,c='Indianred',label='loss:logistic')
plt.plot(sample_name, poisson_user_runtime,c='orange',label='loss:poisson')
plt.legend(['loss:regression','loss:logistic','loss:poisson'])
for a,b in zip(sample_name,ori_user_runtime):
    plt.text(a,b+0.001,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,logistic_user_runtime):
    plt.text(a,b+0.003,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,poisson_user_runtime):
    plt.text(a,b - 0.003,'%.4f'%b,ha='center',va='top',fontsize=6)
plt.title("User Running Time with 3 models --- loss")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Microseconds',fontsize=16)
plt.savefig('./figures_result_loss/loss_user_runtime_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, ori_cpu_time,c='Skyblue',label='loss:regression')
plt.plot(sample_name, logistic_cpu_time,c='Indianred',label='loss:logistic')
plt.plot(sample_name, poisson_cpu_time,c='orange',label='loss:poisson')
plt.legend(['loss:regression','loss:logistic','loss:poisson'])
for a,b in zip(sample_name,ori_cpu_time):
    plt.text(a,b+0.0005,'%.10f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,logistic_cpu_time):
    plt.text(a,b+0.0005,'%.10f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,poisson_cpu_time):
    plt.text(a,b+0.0005,'%.10f'%b,ha='center',va='top',fontsize=6)
plt.title("CPU Time with 3 models --- loss")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_loss/loss_cpu_time_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, ori_RMSE_value,c='Skyblue',label='loss:regression')
plt.plot(sample_name, logistic_RMSE_value,c='Indianred',label='loss:logistic')
plt.plot(sample_name, poisson_RMSE_value,c='orange',label='loss:poisson')
plt.legend(['loss:regression','loss:logistic','loss:poisson'])
for a,b in zip(sample_name,ori_RMSE_value):
    plt.text(a,b+0.01,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,logistic_RMSE_value):
    plt.text(a,b+0.01,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,poisson_RMSE_value):
    plt.text(a,b,'%.4f'%b,ha='center',va='top',fontsize=6)
plt.title("Root Mean Squared Error(RMSE) with 3 models --- loss")
plt.xlabel('Sample',fontsize=16)
plt.savefig('./figures_result_loss/loss_RMSE_plot.png')
plt.show()


    
with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_emb_16.csv","r") as f:
    reader_logistic = csv.reader(f)
    result_list_emb16 = list(reader_logistic)
    emb16_vm_value = []
    emb16_sys_runtime = []
    emb16_user_runtime = []
    emb16_cpu_time = []
    emb16_RMSE_value = []   
    for i in range(2,14):
        emb16_vm_value.append(int(result_list_emb16[1][i]))
    for j in range(1,11):
        emb16_sys_runtime.append(float(result_list_emb16[j][14]))
        emb16_user_runtime.append(float(result_list_emb16[j][15]))
        emb16_cpu_time.append(float(result_list_emb16[j][16]))
        emb16_RMSE_value.append(float(result_list_emb16[j][17]))

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_emb_64.csv","r") as f:
    reader_poisson = csv.reader(f)
    result_list_emb64 = list(reader_poisson)
    emb64_vm_value = []
    emb64_sys_runtime = []
    emb64_user_runtime = []
    emb64_cpu_time = []
    emb64_RMSE_value = []   
    for i in range(2,14):
        emb64_vm_value.append(int(result_list_emb64[1][i]))
    for j in range(1,11):
        emb64_sys_runtime.append(float(result_list_emb64[j][14]))
        emb64_user_runtime.append(float(result_list_emb64[j][15]))
        emb64_cpu_time.append(float(result_list_emb64[j][16]))
        emb64_RMSE_value.append(float(result_list_emb64[j][17]))
        
with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_emb_128.csv","r") as f:
    reader_poisson = csv.reader(f)
    result_list_emb128 = list(reader_poisson)
    emb128_vm_value = []
    emb128_sys_runtime = []
    emb128_user_runtime = []
    emb128_cpu_time = []
    emb128_RMSE_value = []   
    for i in range(2,14):
        emb128_vm_value.append(int(result_list_emb128[1][i]))
    for j in range(1,11):
        emb128_sys_runtime.append(float(result_list_emb128[j][14]))
        emb128_user_runtime.append(float(result_list_emb128[j][15]))
        emb128_cpu_time.append(float(result_list_emb128[j][16]))
        emb128_RMSE_value.append(float(result_list_emb128[j][17]))

plt.figure(dpi=500,figsize=(18,6))
width = 0.22
index = np.arange(len(ori_vm_name))
plt.bar(index - 3*width/2,emb16_vm_value,width,color='Skyblue',label='embedding_dim:16') 
plt.bar(index - width/2,ori_vm_value,width,color='Indianred',label='embedding_dim:32')
plt.bar(index + width/2,emb64_vm_value,width,color='orange',label='embedding_dim:64')
plt.bar(index + 3*width/2,emb128_vm_value,width,color='limegreen',label='embedding_dim:128')
plt.xticks(index,ori_vm_name)
plt.legend(['embedding_dim:16','embedding_dim:32','embedding_dim:64','embedding_dim:128'])
plt.title("The difference of Virtual Memory with 4 models --- embedding dimensions")
for a,b in zip(index - 3*width/2,emb16_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=5)
for a,b in zip(index - width/2,ori_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=5)
for a,b in zip(index + width/2.5,emb64_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=5)
for a,b in zip(index + 1.7*width,emb128_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=5)
plt.ylabel("KB")   
plt.savefig('./figures_result_emb/emb_Vm_bar.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, emb16_sys_runtime,c='Skyblue',label='embedding_dim:16')
plt.plot(sample_name, ori_sys_runtime,c='Indianred',label='embedding_dim:32')
plt.plot(sample_name, emb64_sys_runtime,c='orange',label='embedding_dim:64')
plt.plot(sample_name, emb128_sys_runtime,c='limegreen',label='embedding_dim:128')
plt.legend(['embedding_dim:16','embedding_dim:32','embedding_dim:64','embedding_dim:128'])
for a,b in zip(sample_name,emb16_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,ori_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,emb64_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='right',va='top',fontsize=6)
for a,b in zip(sample_name,emb128_sys_runtime):
    plt.text(a,b+0.005,'%.4f'%b,ha='left',va='bottom',fontsize=6)
plt.title("System Running Time with 4 models - embedding dimensions")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_emb/emb_system_runtime_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, emb16_user_runtime,c='Skyblue',label='embedding_dim:16')
plt.plot(sample_name, ori_user_runtime,c='Indianred',label='embedding_dim:32')
plt.plot(sample_name, emb64_user_runtime,c='orange',label='embedding_dim:64')
plt.plot(sample_name, emb128_user_runtime,c='limegreen',label='embedding_dim:128')
plt.legend(['embedding_dim:16','embedding_dim:32','embedding_dim:64','embedding_dim:128'])
for a,b in zip(sample_name,emb16_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,ori_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,emb64_user_runtime):
    plt.text(a,b,'%.4f'%b,ha='left',va='bottom',fontsize=6)
for a,b in zip(sample_name,emb128_user_runtime):
    plt.text(a,b,'%.4f'%b,ha='right',va='top',fontsize=6)
plt.title("User Running Time with 4 models - embedding dimensions")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Microseconds',fontsize=16)
plt.savefig('./figures_result_emb/emb_user_runtime_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, emb16_cpu_time,c='Skyblue',label='embedding_dim:16')
plt.plot(sample_name, ori_cpu_time,c='Indianred',label='embedding_dim:32')
plt.plot(sample_name, emb64_cpu_time,c='orange',label='embedding_dim:64')
plt.plot(sample_name, emb128_cpu_time,c='limegreen',label='embedding_dim:128')
plt.legend(['embedding_dim:16','embedding_dim:32','embedding_dim:64','embedding_dim:128'])
for a,b in zip(sample_name,emb16_cpu_time):
    plt.text(a,b+0.0005,'%.10f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,ori_cpu_time):
    plt.text(a,b+0.0005,'%.10f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,emb64_cpu_time):
    plt.text(a,b+0.0005,'%.10f'%b,ha='right',va='top',fontsize=6)
for a,b in zip(sample_name,emb128_cpu_time):
    plt.text(a,b+0.005,'%.10f'%b,ha='left',va='bottom',fontsize=6)
plt.title("CPU Time with 4 models - embedding dimensions")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_emb/emb_cpu_time_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, emb16_RMSE_value,c='Skyblue',label='embedding_dim:16')
plt.plot(sample_name, ori_RMSE_value,c='Indianred',label='embedding_dim:32')
plt.plot(sample_name, emb64_RMSE_value,c='orange',label='embedding_dim:64')
plt.plot(sample_name, emb128_RMSE_value,c='limegreen',label='embedding_dim:128')
plt.legend(['embedding_dim:16','embedding_dim:32','embedding_dim:64','embedding_dim:128'])
for a,b in zip(sample_name,emb16_RMSE_value):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,ori_RMSE_value):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,emb64_RMSE_value):
    plt.text(a,b,'%.4f'%b,ha='left',va='bottom',fontsize=6)
for a,b in zip(sample_name,emb128_RMSE_value):
    plt.text(a,b,'%.4f'%b,ha='right',va='top',fontsize=6)
plt.title("Root Mean Squared Error(RMSE) with 4 models - embedding dimensions")
plt.xlabel('Sample',fontsize=16)
plt.savefig('./figures_result_emb/emb_RMSE_plot.png')
plt.show()

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_iter_5.csv","r") as f:
    reader_iter5 = csv.reader(f)
    result_list_iter5 = list(reader_iter5)
    iter5_vm_value = []
    iter5_sys_runtime = []
    iter5_user_runtime = []
    iter5_cpu_time = []
    iter5_RMSE_value = []   
    for i in range(2,14):
        iter5_vm_value.append(int(result_list_iter5[1][i]))
    for j in range(1,11):
        iter5_sys_runtime.append(float(result_list_iter5[j][14]))
        iter5_user_runtime.append(float(result_list_iter5[j][15]))
        iter5_cpu_time.append(float(result_list_iter5[j][16]))
        iter5_RMSE_value.append(float(result_list_iter5[j][17]))

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_iter_20.csv","r") as f:
    reader_iter20 = csv.reader(f)
    result_list_iter20 = list(reader_iter20)
    iter20_vm_value = []
    iter20_sys_runtime = []
    iter20_user_runtime = []
    iter20_cpu_time = []
    iter20_RMSE_value = []   
    for i in range(2,14):
        iter20_vm_value.append(int(result_list_iter20[1][i]))
    for j in range(1,11):
        iter20_sys_runtime.append(float(result_list_iter20[j][14]))
        iter20_user_runtime.append(float(result_list_iter20[j][15]))
        iter20_cpu_time.append(float(result_list_iter20[j][16]))
        iter20_RMSE_value.append(float(result_list_iter20[j][17]))
        
with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_iter_40.csv","r") as f:
    reader_poisson = csv.reader(f)
    result_list_iter40 = list(reader_poisson)
    iter40_vm_value = []
    iter40_sys_runtime = []
    iter40_user_runtime = []
    iter40_cpu_time = []
    iter40_RMSE_value = []   
    for i in range(2,14):
        iter40_vm_value.append(int(result_list_iter40[1][i]))
    for j in range(1,11):
        iter40_sys_runtime.append(float(result_list_iter40[j][14]))
        iter40_user_runtime.append(float(result_list_iter40[j][15]))
        iter40_cpu_time.append(float(result_list_iter40[j][16]))
        iter40_RMSE_value.append(float(result_list_iter40[j][17]))
    
plt.figure(dpi=500,figsize=(18,6))
width = 0.22
index = np.arange(len(ori_vm_name))
plt.bar(index - 3*width/2,iter5_vm_value,width,color='Skyblue',label='iteration:5') 
plt.bar(index - width/2,ori_vm_value,width,color='Indianred',label='iteration:10')
plt.bar(index + width/2,iter20_vm_value,width,color='orange',label='iteration:20')
plt.bar(index + 3*width/2,iter40_vm_value,width,color='limegreen',label='iteration:40')
plt.xticks(index,ori_vm_name)
plt.legend(['iteration:5','iteration:10','iteration:20','iteration:40'])
plt.title("The difference of Virtual Memory with 4 models - iteration")
for a,b in zip(index - 3*width/2,iter5_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index - width/2,ori_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index + width/2,iter20_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index + 3*width/2,iter40_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
plt.ylabel("KB")   
plt.savefig('./figures_result_iter/iter_Vm_bar.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,iter5_sys_runtime,color='Skyblue',label='iteration:5') 
plt.plot(sample_name,ori_sys_runtime,color='Indianred',label='iteration:10')
plt.plot(sample_name,iter20_sys_runtime,color='orange',label='iteration:20')
plt.plot(sample_name,iter40_sys_runtime,color='limegreen',label='iteration:40')
plt.legend(['iteration:5','iteration:10','iteration:20','iteration:40'])
for a,b in zip(sample_name,iter5_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,ori_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,iter20_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,iter40_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("System Running Time with 4 models - iteration")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_iter/iter_system_runtime_plot.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,iter5_user_runtime,color='Skyblue',label='iteration:5') 
plt.plot(sample_name,ori_user_runtime,color='Indianred',label='iteration:10')
plt.plot(sample_name,iter20_user_runtime,color='orange',label='iteration:20')
plt.plot(sample_name,iter40_user_runtime,color='limegreen',label='iteration:40')
plt.legend(['iteration:5','iteration:10','iteration:20','iteration:40'])
for a,b in zip(sample_name,iter5_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,ori_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,iter20_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,iter40_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("User Running Time with 4 models - iteration")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Microseconds',fontsize=16)
plt.savefig('./figures_result_iter/iter_user_runtime_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,iter5_cpu_time,color='Skyblue',label='iteration:5') 
plt.plot(sample_name,ori_cpu_time,color='Indianred',label='iteration:10')
plt.plot(sample_name,iter20_cpu_time,color='orange',label='iteration:20')
plt.plot(sample_name,iter40_cpu_time,color='limegreen',label='iteration:40')
plt.legend(['iteration:5','iteration:10','iteration:20','iteration:40'])
for a,b in zip(sample_name,iter5_cpu_time):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,ori_cpu_time):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,iter20_cpu_time):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,iter40_cpu_time):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("CPU Time with 4 models - iteration")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_iter/iter_cpu_time_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name, iter5_RMSE_value,c='Skyblue',label='iteration:5')
plt.plot(sample_name, ori_RMSE_value,c='Indianred',label='iteration:10')
plt.plot(sample_name, iter20_RMSE_value,c='orange',label='iteration:20')
plt.plot(sample_name, iter40_RMSE_value,c='limegreen',label='iteration:40')
plt.legend(['iteration:5','iteration:10','iteration:20','iteration:40'])
for a,b in zip(sample_name,iter5_RMSE_value):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,ori_RMSE_value):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,iter20_RMSE_value):
    plt.text(a,b,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,iter40_RMSE_value):
    plt.text(a,b,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("Root Mean Squared Error(RMSE) with 4 models - iteration")
plt.xlabel('Sample',fontsize=16)
plt.savefig('./figures_result_iter/iter_RMSE_plot.png')
plt.show()

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_bat_128.csv","r") as f:
    reader = csv.reader(f)
    result_list_bat128 = list(reader)
    bat128_vm_value = []
    bat128_sys_runtime = []
    bat128_user_runtime = []
    bat128_cpu_time = []
    bat128_RMSE_value = []   
    for i in range(2,14):
        bat128_vm_value.append(int(result_list_bat128[1][i]))
    for j in range(1,11):
        bat128_sys_runtime.append(float(result_list_bat128[j][14]))
        bat128_user_runtime.append(float(result_list_bat128[j][15]))
        bat128_cpu_time.append(float(result_list_bat128[j][16]))
        bat128_RMSE_value.append(float(result_list_bat128[j][17]))

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_bat_512.csv","r") as f:
    reader = csv.reader(f)
    result_list_bat512 = list(reader)
    bat512_vm_value = []
    bat512_sys_runtime = []
    bat512_user_runtime = []
    bat512_cpu_time = []
    bat512_RMSE_value = []   
    for i in range(2,14):
        bat512_vm_value.append(int(result_list_bat512[1][i]))
    for j in range(1,11):
        bat512_sys_runtime.append(float(result_list_bat512[j][14]))
        bat512_user_runtime.append(float(result_list_bat512[j][15]))
        bat512_cpu_time.append(float(result_list_bat512[j][16]))
        bat512_RMSE_value.append(float(result_list_bat512[j][17]))
        
with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_bat_1024.csv","r") as f:
    reader = csv.reader(f)
    result_list_bat1024 = list(reader)
    bat1024_vm_value = []
    bat1024_sys_runtime = []
    bat1024_user_runtime = []
    bat1024_cpu_time = []
    bat1024_RMSE_value = []   
    for i in range(2,14):
        bat1024_vm_value.append(int(result_list_bat1024[1][i]))
    for j in range(1,11):
        bat1024_sys_runtime.append(float(result_list_bat1024[j][14]))
        bat1024_user_runtime.append(float(result_list_bat1024[j][15]))
        bat1024_cpu_time.append(float(result_list_bat1024[j][16]))
        bat1024_RMSE_value.append(float(result_list_bat1024[j][17]))
        
plt.figure(dpi=500,figsize=(18,6))
width = 0.22
index = np.arange(len(ori_vm_name))
plt.bar(index - 3*width/2,bat128_vm_value,width,color='Skyblue',label='batch_size:128') 
plt.bar(index - width/2,ori_vm_value,width,color='Indianred',label='batch_size:256')
plt.bar(index + width/2,bat512_vm_value,width,color='orange',label='batch_size:512')
plt.bar(index + 3*width/2,bat1024_vm_value,width,color='limegreen',label='batch_size:1024')
plt.xticks(index,ori_vm_name)
plt.legend(['batch_size:128','batch_size:256','batch_size:512','batch_size:1024'])
plt.title("The difference of Virtual Memory with 4 models - batch size")
for a,b in zip(index - 3*width/2,bat128_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index - width/2,ori_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index + width/2,bat512_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index + 3*width/2,bat1024_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
plt.ylabel("KB")   
plt.savefig('./figures_result_bat/bat_Vm_bar.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,bat128_sys_runtime,color='Skyblue',label='batch_size:128') 
plt.plot(sample_name,ori_sys_runtime,color='Indianred',label='batch_size:256')
plt.plot(sample_name,bat512_sys_runtime,color='orange',label='batch_size:512')
plt.plot(sample_name,bat1024_sys_runtime,color='limegreen',label='batch_size:1024')
plt.legend(['batch_size:128','batch_size:256','batch_size:512','batch_size:1024'])
for a,b in zip(sample_name,bat128_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,ori_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='left',va='bottom',fontsize=6,c='red')
for a,b in zip(sample_name,bat512_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='right',va='bottom',fontsize=6,c='green')
for a,b in zip(sample_name,bat1024_sys_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("System Running Time with 4 models - batch size")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_bat/bat_system_runtime_plot.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,bat128_user_runtime,color='Skyblue',label='batch_size:128') 
plt.plot(sample_name,ori_user_runtime,color='Indianred',label='batch_size:256')
plt.plot(sample_name,bat512_user_runtime,color='orange',label='batch_size:512')
plt.plot(sample_name,bat1024_user_runtime,color='limegreen',label='batch_size:1024')
plt.legend(['batch_size:128','batch_size:256','batch_size:512','batch_size:1024'])
for a,b in zip(sample_name,bat128_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,ori_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,bat512_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,bat1024_user_runtime):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("User Running Time with 4 models - batch size")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Microseconds',fontsize=16)
plt.savefig('./figures_result_bat/bat_user_runtime_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,bat128_cpu_time,color='Skyblue',label='batch_size:128') 
plt.plot(sample_name,ori_cpu_time,color='Indianred',label='batch_size:256')
plt.plot(sample_name,bat512_cpu_time,color='orange',label='batch_size:512')
plt.plot(sample_name,bat1024_cpu_time,color='limegreen',label='batch_size:1024')
plt.legend(['batch_size:128','batch_size:256','batch_size:512','batch_size:1024'])
for a,b in zip(sample_name,bat128_cpu_time):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,ori_cpu_time):
    plt.text(a,b+0.0005,'%.4f'%b,ha='left',va='bottom',fontsize=6,c='red')
for a,b in zip(sample_name,bat512_cpu_time):
    plt.text(a,b+0.0005,'%.4f'%b,ha='right',va='bottom',fontsize=6,c='green')
for a,b in zip(sample_name,bat1024_cpu_time):
    plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("CPU Time with 4 models - batch size")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_bat/bat_cpu_time_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,bat128_RMSE_value,color='Skyblue',label='batch_size:128') 
plt.plot(sample_name,ori_RMSE_value,color='Indianred',label='batch_size:256')
plt.plot(sample_name,bat512_RMSE_value,color='orange',label='batch_size:512')
plt.plot(sample_name,bat1024_RMSE_value,color='limegreen',label='batch_size:1024')
plt.legend(['batch_size:128','batch_size:256','batch_size:512','batch_size:1024'])
for a,b in zip(sample_name,bat128_RMSE_value):
    plt.text(a,b,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,ori_RMSE_value):
    plt.text(a,b-0.001,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,bat512_RMSE_value):
    plt.text(a,b+0.0008,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,bat1024_RMSE_value):
    plt.text(a,b-0.0008,'%.4f'%b,ha='center',va='top',fontsize=6)
plt.title("Root Mean Squared Error(RMSE) with 4 models - batch size")
plt.xlabel('Sample',fontsize=16)
plt.savefig('./figures_result_bat/bat_RMSE_plot.png')
plt.show()

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_l2_02.csv","r") as f:
    reader = csv.reader(f)
    result_list_l202 = list(reader)
    l202_vm_value = []
    l202_sys_runtime = []
    l202_user_runtime = []
    l202_cpu_time = []
    l202_RMSE_value = []   
    for i in range(2,14):
        l202_vm_value.append(int(result_list_l202[1][i]))
    for j in range(1,11):
        l202_sys_runtime.append(float(result_list_l202[j][14]))
        l202_user_runtime.append(float(result_list_l202[j][15]))
        l202_cpu_time.append(float(result_list_l202[j][16]))
        l202_RMSE_value.append(float(result_list_l202[j][17]))

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_l2_04.csv","r") as f:
    reader = csv.reader(f)
    result_list_l204 = list(reader)
    l204_vm_value = []
    l204_sys_runtime = []
    l204_user_runtime = []
    l204_cpu_time = []
    l204_RMSE_value = []   
    for i in range(2,14):
        l204_vm_value.append(int(result_list_l204[1][i]))
    for j in range(1,11):
        l204_sys_runtime.append(float(result_list_l204[j][14]))
        l204_user_runtime.append(float(result_list_l204[j][15]))
        l204_cpu_time.append(float(result_list_l204[j][16]))
        l204_RMSE_value.append(float(result_list_l204[j][17]))
        
with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_l2_08.csv","r") as f:
    reader = csv.reader(f)
    result_list_l208 = list(reader)
    l208_vm_value = []
    l208_sys_runtime = []
    l208_user_runtime = []
    l208_cpu_time = []
    l208_RMSE_value = []   
    for i in range(2,14):
        l208_vm_value.append(int(result_list_l208[1][i]))
    for j in range(1,11):
        l208_sys_runtime.append(float(result_list_l208[j][14]))
        l208_user_runtime.append(float(result_list_l208[j][15]))
        l208_cpu_time.append(float(result_list_l208[j][16]))
        l208_RMSE_value.append(float(result_list_l208[j][17]))
        
plt.figure(dpi=500,figsize=(18,6))
width = 0.22
index = np.arange(len(ori_vm_name))
plt.bar(index - 3*width/2,ori_vm_value,width,color='Skyblue',label='l2_loss:0.0') 
plt.bar(index - width/2,l202_vm_value,width,color='Indianred',label='l2_loss:0.2')
plt.bar(index + width/2,l204_vm_value,width,color='orange',label='l2_loss:0.4')
plt.bar(index + 3*width/2,l208_vm_value,width,color='limegreen',label='l2_loss:0.8')
plt.xticks(index,ori_vm_name)
plt.legend(['l2_loss:0.0','l2_loss:0.2','l2_loss:0.4','l2_loss:0.8'])
plt.title("The difference of Virtual Memory with 4 models - l2 loss penalty")
for a,b in zip(index - 3*width/2,ori_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index - width/2,l202_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index + width/2,l204_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index + 3*width/2,l208_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
plt.ylabel("KB")   
plt.savefig('./figures_result_l2/l2_Vm_bar.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,ori_sys_runtime,color='Skyblue',label='l2_loss:0.0') 
plt.plot(sample_name,l202_sys_runtime,color='Indianred',label='l2_loss:0.2')
plt.plot(sample_name,l204_sys_runtime,color='orange',label='l2_loss:0.4')
plt.plot(sample_name,l208_sys_runtime,color='limegreen',label='l2_loss:0.8')
plt.legend(['l2_loss:0.0','l2_loss:0.2','l2_loss:0.4','l2_loss:0.8'])
# for a,b in zip(sample_name,ori_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
# for a,b in zip(sample_name,l202_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='left',va='bottom',fontsize=6,c='red')
# for a,b in zip(sample_name,l204_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='right',va='bottom',fontsize=6,c='green')
# for a,b in zip(sample_name,l208_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("System Running Time with 4 models - l2 loss penalty")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_l2/l2_system_runtime_plot.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,ori_user_runtime,color='Skyblue',label='l2_loss:0.0') 
plt.plot(sample_name,l202_user_runtime,color='Indianred',label='l2_loss:0.2')
plt.plot(sample_name,l204_user_runtime,color='orange',label='l2_loss:0.4')
plt.plot(sample_name,l208_user_runtime,color='limegreen',label='l2_loss:0.8')
plt.legend(['l2_loss:0.0','l2_loss:0.2','l2_loss:0.4','l2_loss:0.8'])
# for a,b in zip(sample_name,ori_user_runtime):
#     plt.text(a,b,'%.4f'%b,ha='center',va='top',fontsize=6)
# for a,b in zip(sample_name,l202_user_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
# for a,b in zip(sample_name,l204_user_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
# for a,b in zip(sample_name,l208_user_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("User Running Time with 4 models - l2 loss penalty")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Microseconds',fontsize=16)
plt.savefig('./figures_result_l2/l2_user_runtime_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,ori_cpu_time,color='Skyblue',label='l2_loss:0.0') 
plt.plot(sample_name,l202_cpu_time,color='Indianred',label='l2_loss:0.2')
plt.plot(sample_name,l204_cpu_time,color='orange',label='l2_loss:0.4')
plt.plot(sample_name,l208_cpu_time,color='limegreen',label='l2_loss:0.8')
plt.legend(['l2_loss:0.0','l2_loss:0.2','l2_loss:0.4','l2_loss:0.8'])
# for a,b in zip(sample_name,ori_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
# for a,b in zip(sample_name,l202_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='left',va='bottom',fontsize=6,c='red')
# for a,b in zip(sample_name,l204_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='right',va='bottom',fontsize=6,c='green')
# for a,b in zip(sample_name,l208_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("CPU Time with 4 models - l2 loss penalty")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_l2/l2_cpu_time_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,ori_RMSE_value,color='Skyblue',label='l2_loss:0.0') 
plt.plot(sample_name,l202_RMSE_value,color='Indianred',label='l2_loss:0.2')
plt.plot(sample_name,l204_RMSE_value,color='orange',label='l2_loss:0.4')
plt.plot(sample_name,l208_RMSE_value,color='limegreen',label='l2_loss:0.8')
plt.legend(['l2_loss:0.0','l2_loss:0.2','l2_loss:0.4','l2_loss:0.8'])
for a,b in zip(sample_name,ori_RMSE_value):
    plt.text(a,b,'%.4f'%b,ha='center',va='bottom',fontsize=6)
for a,b in zip(sample_name,l202_RMSE_value):
    plt.text(a,b-0.015,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,l204_RMSE_value):
    plt.text(a,b-0.015 ,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,l208_RMSE_value):
    plt.text(a,b,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("Root Mean Squared Error(RMSE) with 4 models - l2 loss penalty")
plt.xlabel('Sample',fontsize=16)
plt.savefig('./figures_result_l2/l2_RMSE_plot.png')
# plt.show()

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_lr_2e.csv","r") as f:
    reader = csv.reader(f)
    result_list_lr2e = list(reader)
    lr2e_vm_value = []
    lr2e_sys_runtime = []
    lr2e_user_runtime = []
    lr2e_cpu_time = []
    lr2e_RMSE_value = []   
    for i in range(2,14):
        lr2e_vm_value.append(int(result_list_lr2e[1][i]))
    for j in range(1,11):
        lr2e_sys_runtime.append(float(result_list_lr2e[j][14]))
        lr2e_user_runtime.append(float(result_list_lr2e[j][15]))
        lr2e_cpu_time.append(float(result_list_lr2e[j][16]))
        lr2e_RMSE_value.append(float(result_list_lr2e[j][17]))

with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_lr_4e.csv","r") as f:
    reader = csv.reader(f)
    result_list_lr4e = list(reader)
    lr4e_vm_value = []
    lr4e_sys_runtime = []
    lr4e_user_runtime = []
    lr4e_cpu_time = []
    lr4e_RMSE_value = []   
    for i in range(2,14):
        lr4e_vm_value.append(int(result_list_lr4e[1][i]))
    for j in range(1,11):
        lr4e_sys_runtime.append(float(result_list_lr4e[j][14]))
        lr4e_user_runtime.append(float(result_list_lr4e[j][15]))
        lr4e_cpu_time.append(float(result_list_lr4e[j][16]))
        lr4e_RMSE_value.append(float(result_list_lr4e[j][17]))
        
with open("/Users/marshallyin/Downloads/Project/spotlight_result/result_lr_8e.csv","r") as f:
    reader = csv.reader(f)
    result_list_lr8e = list(reader)
    lr8e_vm_value = []
    lr8e_sys_runtime = []
    lr8e_user_runtime = []
    lr8e_cpu_time = []
    lr8e_RMSE_value = []   
    for i in range(2,14):
        lr8e_vm_value.append(int(result_list_lr8e[1][i]))
    for j in range(1,11):
        lr8e_sys_runtime.append(float(result_list_lr8e[j][14]))
        lr8e_user_runtime.append(float(result_list_lr8e[j][15]))
        lr8e_cpu_time.append(float(result_list_lr8e[j][16]))
        lr8e_RMSE_value.append(float(result_list_lr8e[j][17]))
        
plt.figure(dpi=500,figsize=(18,6))
width = 0.22
index = np.arange(len(ori_vm_name))
plt.bar(index - 3*width/2,ori_vm_value,width,color='Skyblue',label='learning_rate:1e-2') 
plt.bar(index - width/2,lr2e_vm_value,width,color='Indianred',label='learning_rate:2e-2')
plt.bar(index + width/2,lr4e_vm_value,width,color='orange',label='learning_rate:4e-2')
plt.bar(index + 3*width/2,lr8e_vm_value,width,color='limegreen',label='learning_rate:8e-2')
plt.xticks(index,ori_vm_name)
plt.legend(['learning_rate:1e-2','learning_rate:2e-2','learning_rate:4e-2','learning_rate:8e-2'])
plt.title("The difference of Virtual Memory with 4 models - learning rate")
for a,b in zip(index - 3*width/2,ori_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index - width/2,lr2e_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index + width/2,lr4e_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
for a,b in zip(index + 3*width/2,lr8e_vm_value):
    plt.text(a,b,b,ha='center',va='bottom',fontsize=4.5)
plt.ylabel("KB")   
plt.savefig('./figures_result_lr/lr_Vm_bar.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,ori_sys_runtime,color='Skyblue',label='learning_rate:1e-2') 
plt.plot(sample_name,lr2e_sys_runtime,color='Indianred',label='learning_rate:2e-2')
plt.plot(sample_name,lr4e_sys_runtime,color='orange',label='learning_rate:4e-2')
plt.plot(sample_name,lr8e_sys_runtime,color='limegreen',label='learning_rate:8e-2')
plt.legend(['learning_rate:1e-2','learning_rate:2e-2','learning_rate:4e-2','learning_rate:8e-2'])
# for a,b in zip(sample_name,ori_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
# for a,b in zip(sample_name,l202_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='left',va='bottom',fontsize=6,c='red')
# for a,b in zip(sample_name,l204_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='right',va='bottom',fontsize=6,c='green')
# for a,b in zip(sample_name,l208_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("System Running Time with 4 models - learning rate")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_lr/lr_system_runtime_plot.png')
plt.show()



plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,ori_user_runtime,color='Skyblue',label='learning_rate:1e-2') 
plt.plot(sample_name,lr2e_user_runtime,color='Indianred',label='learning_rate:2e-2')
plt.plot(sample_name,lr4e_user_runtime,color='orange',label='learning_rate:4e-2')
plt.plot(sample_name,lr8e_user_runtime,color='limegreen',label='learning_rate:8e-2')
plt.legend(['learning_rate:1e-2','learning_rate:2e-2','learning_rate:4e-2','learning_rate:8e-2'])
# for a,b in zip(sample_name,ori_user_runtime):
#     plt.text(a,b,'%.4f'%b,ha='center',va='top',fontsize=6)
# for a,b in zip(sample_name,l202_user_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
# for a,b in zip(sample_name,l204_user_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
# for a,b in zip(sample_name,l208_user_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("User Running Time with 4 models - learning rate")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Microseconds',fontsize=16)
plt.savefig('./figures_result_lr/lr_user_runtime_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,ori_cpu_time,color='Skyblue',label='learning_rate:1e-2') 
plt.plot(sample_name,lr2e_cpu_time,color='Indianred',label='learning_rate:2e-2')
plt.plot(sample_name,lr4e_cpu_time,color='orange',label='learning_rate:4e-2')
plt.plot(sample_name,lr8e_cpu_time,color='limegreen',label='learning_rate:8e-2')
plt.legend(['learning_rate:1e-2','learning_rate:2e-2','learning_rate:4e-2','learning_rate:8e-2'])
# for a,b in zip(sample_name,ori_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='top',fontsize=6)
# for a,b in zip(sample_name,l202_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='left',va='bottom',fontsize=6,c='red')
# for a,b in zip(sample_name,l204_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='right',va='bottom',fontsize=6,c='green')
# for a,b in zip(sample_name,l208_sys_runtime):
#     plt.text(a,b+0.0005,'%.4f'%b,ha='center',va='bottom',fontsize=6)
plt.title("CPU Time with 4 models - learning rate")
plt.xlabel('Sample',fontsize=16)
plt.ylabel('Seconds',fontsize=16)
plt.savefig('./figures_result_lr/lr_cpu_time_plot.png')
plt.show()

plt.figure(dpi=500,figsize=(10,6))
plt.plot(sample_name,ori_RMSE_value,color='Skyblue',label='learning_rate:1e-2') 
plt.plot(sample_name,lr2e_RMSE_value,color='Indianred',label='learning_rate:2e-2')
plt.plot(sample_name,lr4e_RMSE_value,color='orange',label='learning_rate:4e-2')
plt.plot(sample_name,lr8e_RMSE_value,color='limegreen',label='learning_rate:8e-2')
plt.legend(['learning_rate:1e-2','learning_rate:2e-2','learning_rate:4e-2','learning_rate:8e-2'])
# for a,b in zip(sample_name,ori_RMSE_value):
#     plt.text(a,b,'%.4f'%b,ha='center',va='bottom',fontsize=6)
# for a,b in zip(sample_name,lr2e_RMSE_value):
#     plt.text(a,b-0.015,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,lr4e_RMSE_value):
    plt.text(a,b-0.015 ,'%.4f'%b,ha='center',va='top',fontsize=6)
for a,b in zip(sample_name,lr8e_RMSE_value):
    plt.text(a,b+0.015,'%.4f'%b,ha='right',va='bottom',fontsize=6)
plt.title("Root Mean Squared Error(RMSE) with 4 models - learning rate")
plt.xlabel('Sample',fontsize=16)
plt.savefig('./figures_result_lr/lr_RMSE_plot.png')
plt.show()