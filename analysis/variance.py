#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import math
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


# In[7]:


data = np.arange(1, 5)
data


# In[4]:


def index_of_dispersion(data):
    return np.var(data)/np.average(data)

avg = 0
for n in data:
    avg += n
avg /= len(data)
var = 0
for n in data:
    var += (n-avg)**2
var /= len(data)

index_of_dispersion(data), var/avg


# In[6]:


def coefficient_of_var(data):
    return math.sqrt(np.var(data))/np.average(data)

avg = 0
for n in data:
    avg += n
avg /= len(data)
var = 0
for n in data:
    var += (n-avg)**2
var /= len(data)

coefficient_of_var(data), math.sqrt(var)/avg


# In[11]:


def quartile_coefficient_of_dispersion(data):
    data = np.sort(data)
    q1 = data[int(len(data)*0.25)]
    q3 = data[int(len(data)*0.75)]
    print(q1, q3)
    return (q3-q1)/(q3+q1)
    
quartile_coefficient_of_dispersion(data)


# In[13]:


def entropy(data, bin_size):
    minn = min(data)
    
    dist = {}
    for n in data:
        bin_num = int((n-minn)/bin_size)
        if bin_num in dist:
            dist[bin_num] += 1
        else:
            dist[bin_num] = 1
    
    ret = 0
    for bin_num in dist:
        p = dist[bin_num]/len(data)
        ret -= p*math.log2(p)
    return ret

entropy(data, 1), math.log2(4)


# In[14]:


def gini(data):
    ret = 0
    for i in range(len(data)-1):
        for j in range(i+1, len(data)):
            ret += abs(data[i]-data[j])
    ret *= 2
    ret /= (2*len(data)**2*np.average(data))
    return ret

gini(data)


# In[25]:


def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
    yarray = cum_wealths / float(sum_wealths)
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return A / (A+B)

gini_coef(data)


# In[26]:


gini(np.array([0, 1])), gini_coef(np.array([0, 1]))


# In[29]:


gini(np.array([-1, 3]))


# ## Anomaly Detection

# In[41]:


X = [[0.3], [0.5], [1], [1.1]]
clf = IsolationForest(random_state=0).fit(X)

data = [[-1.11], [0.1], [0], [90], [100], [10000000]]
clf.predict(data), clf.decision_function(data)


# In[49]:


X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
gm = GaussianMixture(n_components=2, random_state=0).fit(X)
print(gm.means_)
gm.score_samples([[0, 0], [12, 3], [10, 2], [10000, -10000]])


# In[53]:


x = np.array([[0, 0], [12, 3], [10, 2], [10000, -10000]])
scores = gm.score_samples(x)

thresh = np.quantile(scores, .03)
print(thresh)
 
index = np.where(scores <= thresh)
values = x[index]
values


# In[63]:


x = np.array([[0, 0], [12, 3], [10, 2], [10000, -10000]])
nbrs = NearestNeighbors(n_neighbors = 3)
nbrs.fit(x)
distances, indexes = nbrs.kneighbors(x)
print(distances.mean(axis =1))
thresh = np.quantile(distances.mean(axis =1), .5)
print(thresh)
outlier_index = np.where(distances.mean(axis = 1) > thresh)
outlier_index

