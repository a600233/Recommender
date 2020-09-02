#!/usr/bin/env python
# coding: utf-8
from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score,mrr_score,precision_recall_score,sequence_mrr_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
import pandas as pd
import numpy as np
import resource
import os
from collections import defaultdict
from itertools import count
from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split
from sklearn.utils import resample
from multiprocessing import Process, Lock, cpu_count, active_children, Value


ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
movies_df = pd.read_csv("ml-latest-small/movies.csv")

uid_map = defaultdict(count().__next__)
iid_map = defaultdict(count().__next__)
uids = np.array([uid_map[uid] for uid in ratings_df["userId"].values ], dtype=np.int32)
iids = np.array([iid_map[iid] for iid in ratings_df["movieId"].values ], dtype=np.int32)

uid_rev_map = {v: k for k, v in uid_map.items()}
iid_rev_map = {v: k for k, v in iid_map.items()}

ratings = ratings_df["rating"].values.astype(np.float32)
timestamps = ratings_df["timestamp"].values.astype(np.int32)

dataset = Interactions(user_ids=uids,item_ids=iids,ratings=ratings,timestamps=timestamps)



resample_train_cbn = []
resample_test_cbn = []

# prepare bootstrap sample
for i in range(10,0,-1):
    from sklearn.utils import resample
    boot_uid = resample(uids, n_samples=int(uids.size * i / 10), random_state=1)
    boot_iid = resample(iids, n_samples=int(iids.size * i / 10), random_state=1)
#     print(boot_uid.size)
    int_uid = []
    int_iid = []
    int_ratings = []
    int_timestamps = []
    
    temp_sample = ratings_df[ratings_df['userId'].isin(boot_uid)]
    final_sample = temp_sample[temp_sample['movieId'].isin(boot_iid)]
    #print(final_sample)
    int_uid = np.array([uid_map[uid] for uid in final_sample["userId"].values ], dtype=np.int32)
    int_iid = np.array([iid_map[iid] for iid in final_sample["movieId"].values ], dtype=np.int32)
    int_ratings = final_sample['rating'].values.astype(np.float32)
    int_timestamps = final_sample['timestamp'].values.astype(np.int32)

    dataset_boot = Interactions(user_ids=int_uid,item_ids=int_uid,ratings=int_ratings,timestamps=int_timestamps)
    train, test = random_train_test_split(dataset_boot, test_percentage=0.2)
    resample_train_cbn.append(train)
    resample_test_cbn.append(test)


def train_method(num_1,num_2):
    bfr = []
    aft = []
    vminf = []
    
    print(f"PID is {os.getpid()}")
    print("-------------------------before--------------------------------")
    for line in open("/proc/" + str(os.getpid()) + "/status"):
        if line.startswith("Vm"):
            print(line)
            vminf.append(line[:6].replace(":", ""))
            bfr.append(int(line[9:-4]))
            
    model = ExplicitFactorizationModel(n_iter=1)
    intial_time =  resource.getrusage(resource.RUSAGE_SELF); 
    model.fit(resample_train_cbn[num_1])
    final_time = resource.getrusage(resource.RUSAGE_SELF); 
    overall_time_s = final_time.ru_stime - intial_time.ru_stime
    overall_time_u = final_time.ru_utime - intial_time.ru_utime
    print("-------------------------after--------------------------------")
    for line1 in open("/proc/" + str(os.getpid()) + "/status"):
        if line1.startswith("Vm"):
            print(line1)
            aft.append(int(line1[9:-4]))
            
    for i in range(len(bfr)):
        print("The difference between "+vminf[i]+" is "+str(aft[i] - bfr[i]))
    print(f"This process‘s system running time is {overall_time_s}")
    print(f"This process‘s user running time is {overall_time_u}")
    print(f"Root Mean Squared Error is {rmse_score(model, resample_test_cbn[num_2])}")
    print(f"Root Mean Squared Error is {precision_recall_score(model, resample_test_cbn[num_2])}")
    print("----------------------------------------------------------")



# train_method(0,0)

if __name__ == '__main__':
    
    p10 = Process(target=train_method, args=(0,0,))
    p9 = Process(target=train_method, args=(1,1,))
    p8 = Process(target=train_method, args=(2,2,))
    p7 = Process(target=train_method, args=(3,3,))
    p6 = Process(target=train_method, args=(4,4,))
    p5 = Process(target=train_method, args=(5,5,))
    p4 = Process(target=train_method, args=(6,6,))
    p3 = Process(target=train_method, args=(7,7,))
    p2 = Process(target=train_method, args=(8,8,))
    p1 = Process(target=train_method, args=(9,9,))
    
    p10.start()
    p10.join()
    p9.start()
    p9.join()
    p8.start()
    p8.join()
    p7.start()
    p7.join()
    p6.start()
    p6.join()
    p5.start()
    p5.join()
    p4.start()
    p4.join()
    p3.start()
    p3.join()
    p2.start()
    p2.join()
    p1.start()
    p1.join()

      


