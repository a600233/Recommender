#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 09:02:09 2020

@author: marshallyin
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
from decimal import Decimal

path = "/Users/marshallyin/Downloads/Project/spotlight_result"

file_names = os.listdir(path)

file_names.remove('.DS_Store')

for i in file_names:
    file_each = str(i)[:-4]
    with open(f"/Users/marshallyin/Downloads/Project/spotlight_result/{i}","r") as f:
        reader = csv.reader(f)
        result_list = list(reader)
        vm_value = []
        vm_name = []
        for i in range(2,14):
            vm_value.append(int(result_list[1][i]))
            vm_name.append(result_list[0][i])
            
    plt.figure(figsize=(15,6))
    plt.bar(vm_name,vm_value,color='skyblue') 
    plt.title(f"The difference of Virtual Memory {file_each}")
    for a,b in zip(vm_name,vm_value):
        plt.text(a,b,'%.0f'%b,ha='center',va='bottom',fontsize=7)
    plt.ylabel("KB") 
    # if not os.path.exists(f'./result_itself/{file_each}'):
    #     os.makedirs(f'./result_itself/{file_each}')
    plt.savefig(f'./result_itself_div/VM/{file_each}_VM.png')
    plt.show()
    

    
    time_name = []
    sys_runtime = []
    user_runtime = []
    cpu_time = []
    sample_name = []
    RMSE_value = []
    sum_mrr = 0
    avg_mrr = []
    mrr_value = []
    for i in range(14,16):
        time_name.append(result_list[0][i])
    for j in range(1,11):
        sys_runtime.append(float(result_list[j][14]))
        user_runtime.append(float(result_list[j][15]))
        cpu_time.append(float( result_list[j][16]))
        print(cpu_time)
        RMSE_value.append(float(result_list[j][17]))
        sample_name.append(result_list[j][0][:-7])
        
        # for k in range(len(result_list[j][17])):
        #     print(result_list[j][17][k])
        #     sum_mrr += float(result_list[j][17][1:-1][k])
        # avg_mrr.append(sum_mrr/len(result_list[j][17]))
        # print(avg_mrr)
    plt.figure(dpi=500,figsize=(10,6))
    plt.plot(sample_name, sys_runtime,c='red')
    plt.title(f"System Running Time - {file_each}")
    plt.xlabel('Sample',fontsize=16)
    plt.ylabel('Seconds',fontsize=16)
    # if not os.path.exists(f'./result_itself/{file_each}'):
    #     os.makedirs(f'./result_itself/{file_each}')
    plt.savefig(f'./result_itself_div/sys_time/{file_each}_sys_time.png')
    plt.show()
    
    plt.figure(dpi=500,figsize=(10,6))
    plt.plot(sample_name, user_runtime,c='blue')
    plt.title(f"User Running Time - {file_each}")
    plt.xlabel('Sample',fontsize=16)
    plt.ylabel('Microseconds',fontsize=16)
    if not os.path.exists(f'./result_itself/{file_each}'):
        os.makedirs(f'./result_itself/{file_each}')
    plt.savefig(f'./result_itself_div/user_time/{file_each}_user_time.png')
    plt.show()
    
    plt.figure(dpi=500,figsize=(10,6))
    plt.plot(sample_name, cpu_time,c='blue')
    plt.title(f"CPU Time - {file_each}")
    plt.xlabel('Sample',fontsize=16)
    plt.ylabel('Seconds',fontsize=16)
    if not os.path.exists(f'./result_itself/{file_each}'):
        os.makedirs(f'./result_itself/{file_each}')
    plt.savefig(f'./result_itself_div/cpu_time/{file_each}_cpu_time.png')
    plt.show()
    
    plt.figure(dpi=500,figsize=(10,6))
    plt.plot(sample_name, RMSE_value,c='orange')
    plt.title(f"Root Mean Squared Error(RMSE) - {file_each}")
    if not os.path.exists(f'./result_itself/{file_each}'):
        os.makedirs(f'./result_itself/{file_each}')
    plt.savefig(f'./result_itself_div/RMSE_score/{file_each}_RMSE_score.png')
    plt.xlabel('Sample',fontsize=16)
    plt.show()

    # print(time_name)
    # print(sys_runtime)
    # print(user_runtime)
    # print(sample_name)
