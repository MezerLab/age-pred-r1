#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 09:31:43 2019

@author: asier.erramuzpe
"""
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
from scipy.stats import ttest_ind, zscore
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


"""
PCA
"""
"""
R1 + CT - all regions
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')
data_ct = np.delete(data_ct, [20, 33, 38, 50, 52], axis=1)  # 20, 33 (young) 38, 50, 52 (old)
y_ct = np.delete(y_ct, [20, 33, 38, 50, 52]) # 20, 33 (young) 38, 50, 52 (old)
data_ct = zscore(data_ct)
data_ct = np.delete(data_ct, np.where(areas_notdeg2==1), axis=0) 

cortical_parc='volume'
dataset = 'stanford_ms_run1'
cortical_mat_ms1, age_ms1_ct, subs, _ = load_dataset(dataset, 
                                                cortical_parc=cortical_parc,
                                                measure_type='t1')
dataset = 'stanford_ms_run2'
cortical_mat_ms2, age_ms2_ct, subs, _ = load_dataset(dataset, 
                                                cortical_parc=cortical_parc,
                                                measure_type='t1')
cortical_mat_ms1_ct = np.delete(cortical_mat_ms1, 74, axis=0)
cortical_mat_ms2_ct = np.delete(cortical_mat_ms2, 74, axis=0)
cortical_mat_ms1_ct = zscore(cortical_mat_ms1_ct)
cortical_mat_ms2_ct = zscore(cortical_mat_ms2_ct)
cortical_mat_ms1_ct = np.delete(cortical_mat_ms1_ct, np.where(areas_notdeg2==1), axis=0) 
cortical_mat_ms2_ct = np.delete(cortical_mat_ms2_ct, np.where(areas_notdeg2==1), axis=0) 


data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')
data_t1 = np.delete(data_t1, [20, 33, 38, 50, 52], axis=1)  # 20, 33 (young) 38, 50, 52 (old)
y_t1 = np.delete(y_t1, [20, 33, 38, 50, 52]) # 20, 33 (young) 38, 50, 52 (old)
data_t1 = zscore(data_t1)
data_t1 = np.delete(data_t1, np.where(areas_notdeg2==1), axis=0) 

dataset = 'stanford_ms_run1'
cortical_mat_ms1_t1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='r1_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2_t1, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='r1_les')
cortical_mat_ms1_t1 = zscore(cortical_mat_ms1_t1)
cortical_mat_ms2_t1 = zscore(cortical_mat_ms2_t1)
cortical_mat_ms1_t1 = np.delete(cortical_mat_ms1_t1, np.where(areas_notdeg2==1), axis=0) 
cortical_mat_ms2_t1 = np.delete(cortical_mat_ms2_t1, np.where(areas_notdeg2==1), axis=0) 

data_tot = np.concatenate((data_ct, data_t1), axis=0)
cortical_mat_ms1_t1 = np.concatenate((cortical_mat_ms1_ct, cortical_mat_ms1_t1), axis=0)
cortical_mat_ms2_t1 = np.concatenate((cortical_mat_ms2_ct, cortical_mat_ms2_t1), axis=0)


NUM_REPS = 1000
X = data_tot.T



youngsters = np.where(y_t1<=AGE)
y_old = np.delete(y_t1, youngsters)
data_old = np.delete(X, youngsters, axis=0) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_old, features_=np.ones(208), save_plot=False, fig_dpi=50)

adults = np.where(y_t1>AGE)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X, adults, axis=0) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_young, features_=np.ones(208), save_plot=False, fig_dpi=50)

NUM_PCS = 50 # pca_idx
pca = PCA(n_components=NUM_PCS)

lm = linear_model.LinearRegression()
error_old = np.zeros(NUM_REPS)
error_old_lim = np.zeros(NUM_REPS)
error_old_tot = np.empty((data_old.shape[0], NUM_REPS))
error_old_tot[:] = np.nan
indices = np.arange(data_old.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old, idx1, idx2 = train_test_split(data_old, y_old, indices, test_size=0.21)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lm.fit(X_train_pca, y_train)
    y_pred = lm.predict(X_test_pca)
    error_old[idx] = metrics.mean_absolute_error(y_test_old, y_pred)
    error_old_lim[idx] = np.mean(y_test_old - y_pred)
    error_old_tot[idx2,idx] = y_pred
mean_pred_old = np.nanmean(error_old_tot, axis=1)
np.mean(error_old)


X_ms1 = cortical_mat_ms1_t1.T
X_ms2 = cortical_mat_ms2_t1.T

error_ms = np.zeros(NUM_REPS)
error_ms_lim = np.zeros(NUM_REPS)
error_ms1_tot = np.empty((10, NUM_REPS))
error_ms1_tot[:] = np.nan
error_ms2_tot = np.empty((10, NUM_REPS))
error_ms2_tot[:] = np.nan
indices = np.arange(cortical_mat_ms1_t1.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old, test_size=0.21)
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
mean_pred_ms1 = np.nanmean(error_ms1_tot, axis=1)
mean_pred_ms2 = np.nanmean(error_ms2_tot, axis=1)
np.mean(error_ms)

# we want 56 features without reading
NUM_PCS = 56 # pca_idx
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

plt.scatter(np.hstack((mean_pred_ms1)), np.hstack((age_ms1)))
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")

plt.scatter(mean_pred_young, y_young)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")


plt.scatter(np.hstack(( mean_pred_ms2)), np.hstack(( age_ms2)), c='orange', edgecolors='k')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")
plt.xlim(0,100)
plt.title("Age prediction PCA R1") 
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/R1_pca25_mean.eps', format='eps')
plt.close()

sns.distplot(error_young, color='green', label="Young")
sns.distplot(error_old, color='blue', label="Adult")
sns.distplot(error_ms, color='orange', label="MS")
plt.legend()
plt.ylim(0,1.5)
plt.xlim(0,16)
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/R1_pca_errordist_abs.svg', format='svg')
plt.close()

sns.distplot(error_young_lim, color='green', label="Young")
sns.distplot(error_old_lim, color='blue', label="Adult")
sns.distplot(error_ms_lim, color='orange', label="MS")
plt.legend()
plt.ylim(0,0.7)
plt.xlim(-7.5,15)
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/R1_pca_errordist.svg', format='svg')
plt.close()

np.mean(error_young), np.mean(error_old), np.mean(error_ms), np.mean(error_young_lim), np.mean(error_old_lim), np.mean(error_ms_lim)
(3.2424241890538843,
 8.421120217478181,
 9.235654702613955,
 -0.23239398632181837,
 -0.12199140681655658,
 6.960556569820816)

error_young_pca_all = mean_pred_young - y_young
error_old_pca_all = mean_pred_old - y_old
error_ms_pca_r1_1 = mean_pred_ms1
error_ms_pca_r1_2 = mean_pred_ms2


""""
statistics 
""""
from scipy.stats import ttest_ind 

# from 3_all_regions
error_young_pca_r1 
error_old_pca_r1 
error_young_pca_ct
error_old_pca_ct

# from original 
error_young_pca_r1_not_all = error_young_pca_r1
error_old_pca_r1_not_all = error_old_pca_r1
error_young_pca_ct_not_all = error_young_pca_ct
error_old_pca_ct_not_all = error_old_pca_ct


### differences between all regions combined and no all regions
#t, p = ttest_ind(np.abs(error_young_pca_all), np.abs(error_young_pca_r1_not_all))
#p=0.03
#t, p = ttest_ind(np.abs(error_old_pca_all), np.abs(error_old_pca_r1_not_all))
#p=0.36
#t, p = ttest_ind(np.abs(error_young_pca_all), np.abs(error_young_pca_ct_not_all))
#p=0.18
#t, p = ttest_ind(np.abs(error_old_pca_all), np.abs(error_old_pca_ct_not_all))
#p=0.051



#age_ms = [41, 61]
y_old
#y_old[(y_old >= 41) & (y_old <= 61)] # y_old[(y_old >= 40) & (y_old <= 62)]

idx_healthy_old_ms_1 = np.where((y_old >= 39) & (y_old <= 59))
idx_healthy_old_ms_2 = np.where((y_old >= 41) & (y_old <= 61))

# error_young_pca_ct error_young_pca_r1
t, p = ttest_ind(np.abs(error_young_pca_all), np.abs(error_young_pca_r1))
sns.boxplot(data=[np.abs(error_young_pca_all) , np.abs(error_young_pca_r1)], color='g')
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/young_all_r1_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()

# error_old_pca_ct error_old_pca_r1
t, p = ttest_ind(np.abs(error_old_pca_all), np.abs(error_old_pca_r1))
sns.boxplot(data=[np.abs(error_old_pca_all) , np.abs(error_old_pca_r1)], color='b')
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/old_all_r1_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()


# error_young_pca_ct error_young_pca_r1
t, p = ttest_ind(np.abs(error_young_pca_ct), np.abs(error_young_pca_all))
sns.boxplot(data=[np.abs(error_young_pca_ct) , np.abs(error_young_pca_all)], color='g')
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/young_all_ct_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()

# error_old_pca_ct error_old_pca_r1
t, p = ttest_ind(np.abs(error_old_pca_ct), np.abs(error_old_pca_all))
sns.boxplot(data=[np.abs(error_old_pca_ct) , np.abs(error_old_pca_all)], color='b')
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/old_all_ct_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()



#
## error_young_en_ct error_young_en_r1
#t, p = ttest_ind(np.abs(error_young_en_ct), np.abs(error_young_en_r1))
#sns.boxplot(data=[np.abs(error_young_en_ct) , np.abs(error_young_en_r1)], color='g')
#plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1/young_ct_r1_stats_en_' + str(p) + '.svg', format='svg')
#plt.close()
#
## error_old_en_ct error_old_en_r1
#t, p = ttest_ind(np.abs(error_old_en_ct), np.abs(error_old_en_r1))
#sns.boxplot(data=[np.abs(error_old_en_ct) , np.abs(error_old_en_r1)], color='b')
#plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1/old_ct_r1_stats_en_' + str(p) + '.svg', format='svg')
#plt.close()


"""
ms stats
"""

## error_old_pca_ct[idx_healthy_old_ms]
#t, p = ttest_ind(error_old_pca_ct[idx_healthy_old_ms_1], error_ms_pca_ct_1 - y_ms_1)
#sns.boxplot(data=[error_old_pca_ct[idx_healthy_old_ms_1] , error_ms_pca_ct_1 - y_ms_1], color='orange')
#plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1/old_ct_ms1_stats_pca_' + str(p) + '.svg', format='svg')
#plt.close()
## error_old_pca_ct[idx_healthy_old_ms]
#t, p = ttest_ind(error_old_pca_ct[idx_healthy_old_ms_2], error_ms_pca_ct_2 - y_ms_2)
#sns.boxplot(data=[error_old_pca_ct[idx_healthy_old_ms_2] , error_ms_pca_ct_2 - y_ms_2], color='orange')
#plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1/old_ct_ms2_stats_pca_' + str(p) + '.svg', format='svg')
#plt.close()

# error_old_pca_all[idx_healthy_old_ms]
t, p = ttest_ind(error_old_pca_all[idx_healthy_old_ms_1], error_ms_pca_r1_1 - age_ms1) # np.delete(error_ms_pca_r1_1, [1,4]) - np.delete(y_ms_1, [1,4]) # removing outliers, still significant
sns.boxplot(data=[error_old_pca_all[idx_healthy_old_ms_1] , error_ms_pca_r1_1 - age_ms1], color='orange')
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/old_all_ms1_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()
# error_old_pca_all[idx_healthy_old_ms]
t, p = ttest_ind(error_old_pca_all[idx_healthy_old_ms_2], error_ms_pca_r1_2 - age_ms2)
sns.boxplot(data=[error_old_pca_all[idx_healthy_old_ms_2] , error_ms_pca_r1_2 - age_ms2], color='orange')
plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1_104/old_all_ms2_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()

## error_old_en_ct[idx_healthy_old_ms]
#t, p = ttest_ind(error_old_en_ct[idx_healthy_old_ms_1], error_ms_en_ct_1 - y_ms_1)
#sns.boxplot(data=[error_old_en_ct[idx_healthy_old_ms_1] , error_ms_en_ct_1 - y_ms_1], color='orange')
#plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1/old_ct_ms1_stats_en_' + str(p) + '.svg', format='svg')
#plt.close()
## error_old_en_ct[idx_healthy_old_ms]
#t, p = ttest_ind(error_old_en_ct[idx_healthy_old_ms_2], error_ms_en_ct_2 - y_ms_2)
#sns.boxplot(data=[error_old_en_ct[idx_healthy_old_ms_2] , error_ms_en_ct_2 - y_ms_2], color='orange')
#plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1/old_ct_ms2_stats_en_' + str(p) + '.svg', format='svg')
#plt.close()
#
## error_old_en_r1[idx_healthy_old_ms]
#t, p = ttest_ind(error_old_en_r1[idx_healthy_old_ms_1], error_ms_en_r1_1 - y_ms_1) # np.delete(error_ms_en_r1_1, [1,4]) - np.delete(y_ms_1, [1,4]) # removing outliers, still significant
#sns.boxplot(data=[error_old_en_r1[idx_healthy_old_ms_1] , error_ms_en_r1_1 - y_ms_1], color='orange')
#plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1/old_r1_ms1_stats_en_' + str(p) + '.svg', format='svg')
#plt.close()
## error_old_en_r1[idx_healthy_old_ms]
#t, p = ttest_ind(error_old_en_r1[idx_healthy_old_ms_2], error_ms_en_r1_2 - y_ms_2)
#sns.boxplot(data=[error_old_en_r1[idx_healthy_old_ms_2] , error_ms_en_r1_2 - y_ms_2], color='orange')
#plt.savefig('./reports/figures/rebuttal/all_regions_ct_r1/old_r1_ms2_stats_en_' + str(p) + '.svg', format='svg')
#plt.close()


