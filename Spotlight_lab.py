#!/usr/bin/env python
# coding: utf-8

# In[1]:


from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score,mrr_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
import pandas as pd
import numpy as np


# In[2]:


ratings_df = pd.read_csv("ml-latest-small/ratings.csv")
movies_df = pd.read_csv("ml-latest-small/movies.csv")

from collections import defaultdict
from itertools import count

uid_map = defaultdict(count().__next__)
iid_map = defaultdict(count().__next__)
uids = np.array([uid_map[uid] for uid in ratings_df["userId"].values ], dtype=np.int32)
iids = np.array([iid_map[iid] for iid in ratings_df["movieId"].values ], dtype=np.int32)

uid_rev_map = {v: k for k, v in uid_map.items()}
iid_rev_map = {v: k for k, v in iid_map.items()}

ratings = ratings_df["rating"].values.astype(np.float32)
timestamps = ratings_df["timestamp"].values.astype(np.int32)

from spotlight.interactions import Interactions
from spotlight.cross_validation import random_train_test_split

dataset = Interactions(user_ids=uids,item_ids=iids,ratings=ratings,timestamps=timestamps)

#lets initialise the seed, so that its repeatable and reproducible 
train, test = random_train_test_split(dataset, test_percentage=0.2)





import os
print(os.getpid())


# In[8]:


import tracemalloc
import time


#tracemalloc.start()
model = ExplicitFactorizationModel(n_iter=1)
intial_time = time.time()
model.fit(train)
#current, peak = tracemalloc.get_traced_memory()

#print(f"Inital memory usage was {current / 10**6} MB; Peak was {peak / 10**6} MB; and difference is {(peak / 10**6) - (current / 10**6)} MB")
print(f"The total time is {time.time() - intial_time} seconds")

tracemalloc.stop()
#snapshot2 = tracemalloc.take_snapshot()


# In[7]:


#tracemalloc.start()
intial_time = time.time()
print(mrr_score(model, test))
#current, peak = tracemalloc.get_traced_memory()
print(os.getpid())
#print(f"Inital memory usage was {current / 10**6} MB; Peak was {peak / 10**6} MB; and difference is {(peak / 10**6) - (current / 10**6)} MB")
print(f"The total time is {time.time() - intial_time} seconds")


# In[106]:


tracemalloc.start()


tracemalloc.start()
intial_time = time.time()
print(f"Root Mean Squared Error is {rmse_score(model, test)}")
current, peak = tracemalloc.get_traced_memory()

print(f"Inital memory usage was {current / 10**6} MB; Peak was {peak / 10**6} MB; and difference is {(peak / 10**6) - (current / 10**6)} MB")
print(f"The total time is {time.time() - intial_time} seconds")


# In[8]:


# scikit-learn bootstrap
from sklearn.utils import resample
# data sample


# prepare bootstrap sample
for i in range(10,0,-1):
    boot_uid = resample(uids,  n_samples=int(uids.size * i /10), random_state=1)
    boot_iid = resample(iids, n_samples=int(iids.size * i /10), random_state=1)
    boot_ratings = resample(ratings,  n_samples=int(ratings.size * i /10), random_state=1)
    boot_timestamps = resample(timestamps,  n_samples=int(timestamps.size * i /10), random_state=1)
    dataset_boot = Interactions(user_ids=boot_uid,item_ids=boot_iid,ratings=boot_ratings,timestamps=boot_timestamps)

    #lets initialise the seed, so that its repeatable and reproducible 
    train, test = random_train_test_split(dataset_boot, test_percentage=0.2)
    #tracemalloc.start()
    model = ExplicitFactorizationModel(n_iter=1)
    #intial_time = time.time()
    model.fit(train)
    print(f"Root Mean Squared Error is {rmse_score(model, test)}")
    #current, peak = tracemalloc.get_traced_memory()

    #print(f"Inital memory usage was {current / 10**6} MB; Peak was {peak / 10**6} MB; and difference is {(peak / 10**6) - (current / 10**6)} MB")
    #print(f"The total time is {time.time() - intial_time} seconds")

    #tracemalloc.stop()


    
    





# In[ ]:





# In[ ]:




