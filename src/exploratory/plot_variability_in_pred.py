#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:32:02 2020

@author: asier.erramuzpe
"""

###
### Calculate GAP and deviation
###

import numpy as np
from src.data.make_dataset import (load_cort,
                                   load_dataset,
                                   load_cort_areas,
                                   )

from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     )
from sklearn import metrics
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

areas = load_cort_areas()


AGE = 25
NUM_AREAS = 148
areas_notdeg2_idx = [  0,   1,   4,   5,  11,  12,  13,  17,  20,  21,  22,  23,  30,  31,  33,
                       34,  36,  38,  41,  42,  46,  47,  49,  61,  62,  63,  69, 71, 74,
                       75,  79,  94,  95,  96,  97, 104, 105, 108, 115, 116, 120, 136, 137,
                       143] 
areas_notdeg2 = np.zeros(NUM_AREAS)
areas_notdeg2[areas_notdeg2_idx] = 1

def smooth(y, box_pts):
    # based on a moving average box (by convolution)
    # https://stackoverflow.com/a/26337730
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


"""
PCA
"""
"""
R1
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')

dataset = 'stanford_ms_run1'
cortical_mat_ms1_t1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='r1_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2_t1, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='r1_les')

NUM_REPS = 1000
X = data_t1.T

youngsters = np.where(y_t1<=AGE)
y_old = np.delete(y_t1, youngsters)
data_old = np.delete(X, youngsters, axis=0) 
data_old = np.delete(data_old, np.where(areas_notdeg2==1), axis=1) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_old, features_=np.delete(areas, np.where(areas_notdeg2==1)), save_plot=False, fig_dpi=50)

adults = np.where(y_t1>AGE)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X, adults, axis=0) 
data_young = np.delete(data_young, np.where(areas_notdeg2==1), axis=1) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_young, features_=np.delete(areas, np.where(areas_notdeg2==1)), save_plot=False, fig_dpi=50)

NUM_PCS = 29 # pca_idx
pca = PCA(n_components=NUM_PCS)

lm = linear_model.LinearRegression()
error_old = np.zeros(NUM_REPS)
error_old_lim = np.zeros(NUM_REPS)
error_old_tot = np.empty((data_old.shape[0], NUM_REPS))
error_old_tot[:] = np.nan
var_tot = np.empty((data_old.shape[0], NUM_REPS))
var_tot[:] = np.nan
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
    var_tot[idx2,idx] = y_test_old - y_pred

mean_pred_old = np.nanmean(error_old_tot, axis=1)
np.mean(error_old)


X_ms1 = cortical_mat_ms1_t1.T
X_ms2 = cortical_mat_ms2_t1.T
X_ms1 = np.delete(X_ms1, np.where(areas_notdeg2==1), axis=1) 
X_ms2 = np.delete(X_ms2, np.where(areas_notdeg2==1), axis=1) 

error_ms = np.zeros(NUM_REPS)
error_ms_lim = np.zeros(NUM_REPS)
error_ms1_tot = np.empty((10, NUM_REPS))
error_ms1_tot[:] = np.nan
error_ms2_tot = np.empty((10, NUM_REPS))
error_ms2_tot[:] = np.nan
var_tot1 = np.empty((10, NUM_REPS))
var_tot1[:] = np.nan
var_tot2 = np.empty((10, NUM_REPS))
var_tot2[:] = np.nan
indices = np.arange(cortical_mat_ms1_t1.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca_ms1 = pca.transform(X_ms1)
    X_test_pca_ms2 = pca.transform(X_ms2)
    lm.fit(X_train_pca, y_train)
    y_pred_ms1 = lm.predict(X_test_pca_ms1)
    y_pred_ms2 = lm.predict(X_test_pca_ms2)
    error_ms[idx] = np.mean((metrics.mean_absolute_error(y_pred_ms1, age_ms1), metrics.mean_absolute_error(y_pred_ms2, age_ms2)))
    error_ms_lim[idx] = np.mean((np.mean(y_pred_ms1 - age_ms1), np.mean(y_pred_ms2 - age_ms2)))
    error_ms1_tot[:,idx] = y_pred_ms1
    error_ms2_tot[:,idx] = y_pred_ms2
    var_tot1[:,idx] = y_pred_ms1 - age_ms1
    var_tot2[:,idx] = y_pred_ms2 - age_ms2
mean_pred_ms1 = np.nanmean(error_ms1_tot, axis=1)
mean_pred_ms2 = np.nanmean(error_ms2_tot, axis=1)
np.mean(error_ms)

NUM_PCS = 28 # pca_idx
pca = PCA(n_components=NUM_PCS)
error_young = np.zeros(NUM_REPS)
error_young_lim = np.zeros(NUM_REPS)
error_young_tot = np.empty((data_young.shape[0], NUM_REPS))
error_young_tot[:] = np.nan
var_tot_young = np.empty((data_young.shape[0], NUM_REPS))
var_tot_young[:] = np.nan
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
    var_tot_young[idx2,idx] = y_test_young - y_pred
 
mean_pred_young = np.nanmean(error_young_tot, axis=1)
np.mean(error_young)


plt.errorbar(mean_pred_old, y_old, xerr= np.nanstd(error_old_tot, axis=1), fmt='o')
plt.plot(range(15,100),range(15,100))

plt.errorbar(mean_pred_young, y_young, xerr= np.nanstd(var_tot_young, axis=1), fmt='o')
plt.plot(range(15,100),range(15,100))

plt.errorbar(mean_pred_ms1, age_ms1, xerr= np.nanstd(var_tot1, axis=1), fmt='o',  c='orange')
plt.plot(range(15,100),range(15,100))

plt.errorbar(mean_pred_ms2, age_ms2, xerr= np.nanstd(var_tot2, axis=1), fmt='o',  c='orange', ecolor='black')
plt.plot(range(15,100),range(15,100))
plt.xlim(0,100)
plt.savefig('./reports/poster_ohbm/R1_pca25_mean_var.eps', format='eps')
plt.close()

plt.plot(smooth(mean_pred_old[np.argsort(y_old)], 2), y_old[np.argsort(y_old)])
plt.plot(range(15,100),range(15,100))
plt.plot(smooth(mean_pred_young[np.argsort(y_young)], 2), y_young[np.argsort(y_young)])
plt.plot(range(15,100),range(15,100))
plt.savefig('./reports/poster_ohbm/R1_pca25_gap_bar.eps', format='eps')
plt.close()


"""
PCA
"""
"""
CT 
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')
cortical_parc='volume'
dataset = 'stanford_ms_run1'
cortical_mat_ms1, age_ms, subs, _ = load_dataset(dataset, 
                                                cortical_parc=cortical_parc,
                                                measure_type='t1')
dataset = 'stanford_ms_run2'
cortical_mat_ms2, age_ms, subs, _ = load_dataset(dataset, 
                                                cortical_parc=cortical_parc,
                                                measure_type='t1')
cortical_mat_ms1_ct = np.delete(cortical_mat_ms1, 74, axis=0)
cortical_mat_ms2_ct = np.delete(cortical_mat_ms2, 74, axis=0)

NUM_REPS = 1000
NUM_PCS = 37
pca = PCA(n_components=NUM_PCS)

X = data_ct.T

youngsters = np.where(y_ct<=AGE)
y_old = np.delete(y_ct, youngsters)
data_old = np.delete(X, youngsters, axis=0) 
data_old = np.delete(data_old, np.where(areas_notdeg2==1), axis=1) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_old, features_=np.delete(areas, np.where(areas_notdeg2==1)), save_plot=False, fig_dpi=50)

adults = np.where(y_ct>AGE)
y_young = np.delete(y_ct, adults)
data_young = np.delete(X, adults, axis=0) 
data_young = np.delete(data_young, np.where(areas_notdeg2==1), axis=1) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_young, features_=np.delete(areas, np.where(areas_notdeg2==1)), save_plot=False, fig_dpi=50)

lm = linear_model.LinearRegression()
error_old = np.zeros(NUM_REPS)
error_old_tot = np.empty((data_old.shape[0], NUM_REPS))
error_old_tot[:] = np.nan
error_old_lim = np.zeros(NUM_REPS)
var_tot = np.empty((data_old.shape[0], NUM_REPS))
var_tot[:] = np.nan
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
    var_tot[idx2,idx] = y_test_old - y_pred

mean_pred_old = np.nanmean(error_old_tot, axis=1)
np.mean(error_old)


X_ms1 = cortical_mat_ms1_ct.T
X_ms2 = cortical_mat_ms2_ct.T
X_ms1 = np.delete(X_ms1, np.where(areas_notdeg2==1), axis=1) 
X_ms2 = np.delete(X_ms2, np.where(areas_notdeg2==1), axis=1) 

error_ms = np.zeros(NUM_REPS)
error_ms1_tot = np.empty((10, NUM_REPS))
error_ms1_tot[:] = np.nan
error_ms2_tot = np.empty((10, NUM_REPS))
error_ms2_tot[:] = np.nan
error_ms_lim = np.zeros(NUM_REPS)
var_tot1 = np.empty((10, NUM_REPS))
var_tot1[:] = np.nan
var_tot2 = np.empty((10, NUM_REPS))
var_tot2[:] = np.nan
indices = np.arange(cortical_mat_ms1_ct.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca_ms1 = pca.transform(X_ms1)
    X_test_pca_ms2 = pca.transform(X_ms2)
    lm.fit(X_train_pca, y_train)
    y_pred_ms1 = lm.predict(X_test_pca_ms1)
    y_pred_ms2 = lm.predict(X_test_pca_ms2)
    error_ms[idx] = np.mean((metrics.mean_absolute_error(y_pred_ms1, age_ms1), metrics.mean_absolute_error(y_pred_ms2, age_ms2)))
    error_ms_lim[idx] = np.mean((np.mean(y_pred_ms1 - age_ms1), np.mean(y_pred_ms2 - age_ms2)))
    error_ms1_tot[:,idx] = y_pred_ms1
    error_ms2_tot[:,idx] = y_pred_ms2
    var_tot1[:,idx] = y_pred_ms1 - age_ms1
    var_tot2[:,idx] = y_pred_ms2 - age_ms2

mean_pred_ms1 = np.nanmean(error_ms1_tot, axis=1)
mean_pred_ms2 = np.nanmean(error_ms2_tot, axis=1)
np.mean(error_ms)

NUM_PCS = 45
pca = PCA(n_components=NUM_PCS)
error_young = np.zeros(NUM_REPS)
error_young_tot = np.empty((data_young.shape[0], NUM_REPS))
error_young_tot[:] = np.nan
error_young_lim = np.zeros(NUM_REPS)
var_tot_young = np.empty((data_young.shape[0], NUM_REPS))
var_tot_young[:] = np.nan
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
    var_tot_young[idx2,idx] = y_test_young - y_pred
mean_pred_young = np.nanmean(error_young_tot, axis=1)
np.mean(error_young)


plt.errorbar(mean_pred_old, y_old, xerr= np.nanstd(var_tot, axis=1), fmt='o')
plt.plot(range(15,100),range(15,100))

plt.errorbar(mean_pred_young, y_young, xerr= np.nanstd(var_tot_young, axis=1), fmt='o')
plt.plot(range(15,100),range(15,100))

plt.errorbar(mean_pred_ms1, age_ms1, xerr= np.nanstd(var_tot1, axis=1), fmt='o',  c='orange')
plt.plot(range(15,100),range(15,100))

plt.errorbar(mean_pred_ms2, age_ms2, xerr= np.nanstd(var_tot2, axis=1), fmt='o',  c='orange', ecolor='black')
plt.plot(range(15,100),range(15,100))
plt.xlim(0,100)
plt.savefig('./reports/poster_ohbm/CT_pca25_mean_var.eps', format='eps')
plt.close()


plt.plot(smooth(mean_pred_old[np.argsort(y_old)], 2), y_old[np.argsort(y_old)])
plt.plot(range(15,100),range(15,100))
plt.plot(smooth(mean_pred_young[np.argsort(y_young)], 2), y_young[np.argsort(y_young)])
plt.plot(range(15,100),range(15,100))
plt.savefig('./reports/poster_ohbm/CT_pca25_gap_bar.eps', format='eps')
plt.close()
