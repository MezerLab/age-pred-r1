#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:01:17 2020

@author: asier.erramuzpe
"""

import numpy as np
import matplotlib.pyplot as plt
import pwlf
from pyDOE import lhs


data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')

X = data_t1.T
X = np.delete(X, np.where(areas_notdeg2==1), axis=1) 

n_segments = 2
degree = 1
NUM_REPS = 10

error_old = np.zeros(NUM_REPS)
error_old_lim = np.zeros(NUM_REPS)
error_old_tot = np.empty((X.shape[0], NUM_REPS))
error_old_tot[:] = np.nan
indices = np.arange(X.shape[0])
for idx in range(NUM_REPS):
    print(idx)
    X_train, X_test, y_train, y_test_old, idx1, idx2 = train_test_split(X, y_t1, indices)
    my_mv_model = pwlf.PiecewiseMultivariate(X_train, y_train, n_segments, degree)
    my_mv_model.fit()
    y_pred = my_mv_model.predict(X_test)
    error_old[idx] = metrics.mean_absolute_error(y_test_old, y_pred)
    error_old_lim[idx] = np.mean(y_test_old - y_pred)
    error_old_tot[idx2,idx] = y_pred
mean_pred_old = np.nanmean(error_old_tot, axis=1)
#mean_pred_old[np.where(mean_pred_old<0)] = 0
#mean_pred_old[np.where(mean_pred_old>100)] = 0
#np.nanmean(mean_pred_old)

plt.scatter(mean_pred_old, y_t1)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")
plt.xlim(0,100)
plt.savefig('./reports/figures/rebuttal/piecewise.svg', format='svg')

# 3seg 2deg - 24.25
# 3seg 1deg - 24.55
# 2seg 2deg - 27.26
# 2seg 1 deg - 24


"""
Our approach
"""

mean_min_t1_age #take it from lifepan_rate_gm_all

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')

X = data_t1.T
X = np.delete(X, np.where(areas_notdeg2==1), axis=1) 
mean_min_t1_age = np.delete(mean_min_t1_age, np.where(areas_notdeg2==1)) 

median = np.median(mean_min_t1_age) # 45 is the median peak

# not including in the 45s (we are deleting them)
areas_peak_below = np.where(mean_min_t1_age<median)[0] # 46 areas
areas_peak_over = np.where(mean_min_t1_age>median)[0] # 45 areas

X_below = np.delete(X, areas_peak_over, axis=1) 
X_over = np.delete(X, areas_peak_below, axis=1) 

youngsters = np.where(y_t1<=median)
y_old = np.delete(y_t1, youngsters)
data_old = np.delete(X_below, youngsters, axis=0) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_old, features_=np.ones(data_old.shape[1]), save_plot=False, fig_dpi=50)

adults = np.where(y_t1>median)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X_over, adults, axis=0) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_young, features_=np.ones(data_young.shape[1]), save_plot=False, fig_dpi=50)

NUM_PCS = 15 # pca_idx
pca = PCA(n_components=NUM_PCS)

lm = linear_model.LinearRegression()
error_old = np.zeros(NUM_REPS)
error_old_lim = np.zeros(NUM_REPS)
error_old_tot = np.empty((data_old.shape[0], NUM_REPS))
error_old_tot[:] = np.nan
indices = np.arange(data_old.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old, idx1, idx2 = train_test_split(data_old, y_old, indices)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lm.fit(X_train_pca, y_train)
    y_pred = lm.predict(X_test_pca)
    error_old[idx] = metrics.mean_absolute_error(y_test_old, y_pred)
    error_old_lim[idx] = np.mean(y_test_old - y_pred)
    error_old_tot[idx2,idx] = y_pred
mean_pred_old = np.nanmean(error_old_tot, axis=1)
np.mean(error_old)

# we want 27 features without reading
NUM_PCS = 24 # pca_idx
pca = PCA(n_components=NUM_PCS)
error_young = np.zeros(NUM_REPS)
error_young_lim = np.zeros(NUM_REPS)
error_young_tot = np.empty((data_young.shape[0], NUM_REPS))
error_young_tot[:] = np.nan
indices = np.arange(data_young.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young, idx1, idx2 = train_test_split(data_young, y_young, indices)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lm.fit(X_train_pca, y_train)
    y_pred = lm.predict(X_test_pca)
    error_young[idx] = metrics.mean_absolute_error(y_test_young, y_pred)
    error_young_lim[idx] = np.mean(y_test_young - y_pred)
    error_young_tot[idx2,idx] = y_pred
mean_pred_young = np.nanmean(error_young_tot, axis=1)
np.mean(error_young)

plt.scatter(mean_pred_old, y_old)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")


plt.scatter(mean_pred_young, y_young, c='green')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")
plt.xlim(0,100)
plt.title("Age prediction PCA R1") 
plt.savefig('./reports/figures/rebuttal/piecewise_our.svg', format='svg')
plt.close()

np.mean(error_young), np.mean(error_old)
5.89151757102244, 9.208006086991238