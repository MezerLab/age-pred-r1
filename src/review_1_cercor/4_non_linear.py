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
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

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
R1 - SVR
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')
data_t1 = zscore(data_t1)

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

NUM_PCS = 30 # pca_idx
pca = PCA(n_components=NUM_PCS)

lm = SVR(kernel='linear') # linear, poly, rbf,
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
NUM_PCS = 27 # pca_idx
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

plt.scatter(mean_pred_young, y_young, c='g')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")
plt.xlim(0,100)
plt.title("Age prediction PCA R1") 
plt.savefig('./reports/figures/rebuttal/non_linear/SVR_linear.eps', format='eps')
plt.close()

error_young_pca_r1_svr = mean_pred_young - y_young
error_old_pca_r1_svr = mean_pred_old - y_old

np.mean(error_young), np.mean(error_old)
#(3.9609219591446774, 9.877936926806628)

"""
CT - SVR
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')
data_ct = zscore(data_ct)

NUM_REPS = 1000
NUM_PCS = 38
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

lm = SVR(kernel='linear') # linear, poly, rbf,
error_old = np.zeros(NUM_REPS)
error_old_tot = np.empty((data_old.shape[0], NUM_REPS))
error_old_tot[:] = np.nan
error_old_lim = np.zeros(NUM_REPS)
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

NUM_PCS = 43
pca = PCA(n_components=NUM_PCS)
error_young = np.zeros(NUM_REPS)
error_young_tot = np.empty((data_young.shape[0], NUM_REPS))
error_young_tot[:] = np.nan
error_young_lim = np.zeros(NUM_REPS)
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

plt.scatter(mean_pred_young, y_young, c='g')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")
plt.title("Age prediction PCA CT") 
plt.savefig('./reports/figures/rebuttal/non_linear/SVR_linear_ct.eps', format='eps')


error_young_pca_ct_svr = mean_pred_young - y_young
error_old_pca_ct_svr = mean_pred_old - y_old

np.mean(error_young), np.mean(error_old)
#(3.3729686235279828, 10.009140310737035)
""""
statistics 
""""
from scipy.stats import ttest_ind 


# linear vs SVR r1
t, p = ttest_ind(np.abs(error_young_pca_r1), np.abs(error_young_pca_r1_svr))
sns.boxplot(data=[np.abs(error_young_pca_r1) , np.abs(error_young_pca_r1_svr)], color='g')
plt.savefig('./reports/figures/rebuttal/non_linear/linear_vs_svr_r1_young_' + str(p) + '.svg', format='svg')
plt.close()

# linear vs SVR r1
t, p = ttest_ind(np.abs(error_old_pca_r1), np.abs(error_old_pca_r1_svr))
sns.boxplot(data=[np.abs(error_old_pca_r1) , np.abs(error_old_pca_r1_svr)], color='b')
plt.savefig('./reports/figures/rebuttal/non_linear/linear_vs_svr_r1_old_' + str(p) + '.svg', format='svg')
plt.close()

# linear vs SVR ct
t, p = ttest_ind(np.abs(error_young_pca_ct_svr), np.abs(error_young_pca_ct))
sns.boxplot(data=[np.abs(error_young_pca_ct_svr) , np.abs(error_young_pca_ct)], color='g')
plt.savefig('./reports/figures/rebuttal/non_linear/linear_vs_svr_ct_young_' + str(p) + '.svg', format='svg')
plt.close()

# linear vs SVR ct
t, p = ttest_ind(np.abs(error_old_pca_ct_svr), np.abs(error_old_pca_ct))
sns.boxplot(data=[np.abs(error_old_pca_ct_svr) , np.abs(error_old_pca_ct)], color='b')
plt.savefig('./reports/figures/rebuttal/non_linear/linear_vs_svr_ct_old_' + str(p) + '.svg', format='svg')
plt.close()







"""
R1 - KernelRidgeReg no groups
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')
data_t1 = zscore(data_t1)

NUM_REPS = 1000
X = data_t1.T

y_old = y_t1
data_old = X
data_old = np.delete(data_old, np.where(areas_notdeg2==1), axis=1) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_old, features_=np.delete(areas, np.where(areas_notdeg2==1)), save_plot=False, fig_dpi=50)

NUM_PCS = 47 # pca_idx
pca = PCA(n_components=NUM_PCS)

lm = KernelRidge(kernel='poly', degree=2) # linear, poly, rbf,
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
#9.786671218705714

plt.scatter(mean_pred_old, y_old)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")
plt.savefig('./reports/figures/rebuttal/non_linear/KR_non_linear_all.eps', format='eps')
plt.close()

error_old_pca_r1_kr = mean_pred_old - y_old


"""
CT - KernelRidgeReg no groups
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')
data_ct = zscore(data_ct)

NUM_REPS = 1000
X = data_ct.T

y_old = y_ct
data_old = X
data_old = np.delete(data_old, np.where(areas_notdeg2==1), axis=1) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_old, features_=np.delete(areas, np.where(areas_notdeg2==1)), save_plot=False, fig_dpi=50)

NUM_PCS = 49 # pca_idx
pca = PCA(n_components=NUM_PCS)

lm = KernelRidge(kernel='poly', degree=2) # linear, poly, rbf,
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
#9.118353306745935
plt.scatter(mean_pred_old, y_old)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")
plt.savefig('./reports/figures/rebuttal/non_linear/KR_non_linear_all_ct.eps', format='eps')
plt.close()

error_old_pca_ct_kr = mean_pred_old - y_old


