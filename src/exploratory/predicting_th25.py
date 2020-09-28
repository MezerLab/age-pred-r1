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
R1 - ElasticNet
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


#prepare a range of parameters to test
alphas = np.array([1,]) 
l1_ratio=np.array([0.9])
#create and fit a ridge regression model, testing each alpha
model = linear_model.ElasticNet() #We have chosen to just normalize the data by default, you could GridsearchCV this is you wanted
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas, l1_ratio=l1_ratio))
grid.fit(data_old, y_old)
# summarize the results of the grid search

# we want 30 features
grid.best_estimator_.alpha=0.01
grid.best_estimator_.l1_ratio=0.9999
lm = linear_model.ElasticNet(alpha=grid.best_estimator_.alpha, l1_ratio=grid.best_estimator_.l1_ratio)
error_old = np.zeros(NUM_REPS)
error_old_lim = np.zeros(NUM_REPS)
error_old_tot = np.empty((data_old.shape[0],NUM_REPS))
error_old_tot[:] = np.nan
indices = np.arange(data_old.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old, idx1, idx2 = train_test_split(data_old, y_old, indices)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    error_old[idx] = metrics.mean_absolute_error(y_test_old, y_pred)
    error_old_tot[idx2,idx] = y_pred
    error_old_lim[idx] = np.mean(y_test_old - y_pred)
mean_pred_old = np.nanmean(error_old_tot, axis=1)
np.mean(error_old)
#print(np.count_nonzero(lm.coef_))
#
#plt.scatter(mean_pred_old, y_old)
#plt.plot(range(15,100),range(15,100))
#plt.xlabel("predicted age")
#plt.ylabel("real age")

cortical_mat_ms1_t1 = np.delete(cortical_mat_ms1_t1, np.where(areas_notdeg2==1), axis=0) 
cortical_mat_ms2_t1 = np.delete(cortical_mat_ms2_t1, np.where(areas_notdeg2==1), axis=0) 

error_ms = np.zeros(NUM_REPS)
error_ms_lim = np.zeros(NUM_REPS)
error_ms1_tot = np.empty((cortical_mat_ms1_t1.shape[1],NUM_REPS))
error_ms1_tot[:] = np.nan
error_ms2_tot = np.empty((cortical_mat_ms2_t1.shape[1],NUM_REPS))
error_ms2_tot[:] = np.nan
indices = np.arange(cortical_mat_ms1_t1.shape[1])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    y_pred_ms1 = lm.predict(cortical_mat_ms1_t1.T)
    y_pred_ms2 = lm.predict(cortical_mat_ms2_t1.T)
    error_ms[idx] = np.mean((metrics.mean_absolute_error(y_pred_ms1, age_ms1), metrics.mean_absolute_error(y_pred_ms2, age_ms2)))
    error_ms_lim[idx] = np.mean((np.mean(y_pred_ms1 - age_ms1), np.mean(y_pred_ms2 - age_ms2)))
    error_ms1_tot[:,idx] = y_pred_ms1
    error_ms2_tot[:,idx] = y_pred_ms2
mean_pred_ms1 = np.nanmean(error_ms1_tot, axis=1)
mean_pred_ms2 = np.nanmean(error_ms2_tot, axis=1)
np.mean(error_ms)


# we want 28 features
# we want 27 features without reading
grid.best_estimator_.alpha=0.0055
grid.best_estimator_.l1_ratio=0.999
lm = linear_model.ElasticNet(alpha=grid.best_estimator_.alpha, l1_ratio=grid.best_estimator_.l1_ratio)
error_young = np.zeros(NUM_REPS)
error_young_lim = np.zeros(NUM_REPS)
error_young_tot = np.empty((data_young.shape[0],NUM_REPS))
error_young_tot[:] = np.nan
indices = np.arange(data_young.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young, idx1, idx2 = train_test_split(data_young, y_young, indices)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    error_young[idx] = metrics.mean_absolute_error(y_test_young, y_pred)
    error_young_lim[idx] = np.mean(y_test_young - y_pred)
    error_young_tot[idx2,idx] = y_pred
mean_pred_young = np.nanmean(error_young_tot, axis=1)
#print(np.mean(error_young))
#print(np.count_nonzero(lm.coef_))
#
#plt.scatter(mean_pred_young, y_young)
#plt.plot(range(15,100),range(15,100))
#plt.xlabel("predicted age")
#plt.ylabel("real age")


plt.scatter(mean_pred_old, y_old)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")

plt.scatter(np.hstack((mean_pred_ms1)), np.hstack((age_ms1)))
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")
plt.title("Age prediction ElasticNet")  

plt.scatter(mean_pred_young, y_young)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")

plt.scatter(np.hstack(( mean_pred_ms2)), np.hstack(( age_ms2)), c='orange', edgecolors='k')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")
plt.xlim(0,100)
plt.title("Age prediction ElasticNet R1") 
plt.savefig('./reports/poster_ohbm/R1_elastic_net25_mean.eps', format='eps')


sns.distplot(error_young, color='green', label="Young")
sns.distplot(error_old, color='blue', label="Adult")
sns.distplot(error_ms, color='orange', label="MS")
plt.legend()
plt.ylim(0,1.5)
plt.xlim(0,16)
plt.savefig('./reports/poster_ohbm/R1_en_errordist_abs.svg', format='svg')
plt.close()

sns.distplot(error_young_lim, color='green', label="Young")
sns.distplot(error_old_lim, color='blue', label="Adult")
sns.distplot(error_ms_lim, color='orange', label="MS")
plt.legend()
plt.ylim(0,0.7)
plt.xlim(-7.5,12.5)
plt.savefig('./reports/poster_ohbm/R1_en_errordist.svg', format='svg')
plt.close()
np.mean(error_young), np.mean(error_old), np.mean(error_ms), np.mean(error_young_lim), np.mean(error_old_lim), np.mean(error_ms_lim)
(3.5830963392603907,
 10.215614095291377,
 12.550530993631048,
 -0.08900306041328757,
 0.6530967752329196,
 8.136576455742453)

error_young_en_r1 = mean_pred_young - y_young
error_old_en_r1 = mean_pred_old - y_old
error_ms_en_r1_1 = mean_pred_ms1
error_ms_en_r1_2 = mean_pred_ms2
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

NUM_PCS = 30 # pca_idx
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
mean_pred_ms1 = np.nanmean(error_ms1_tot, axis=1)
mean_pred_ms2 = np.nanmean(error_ms2_tot, axis=1)
np.mean(error_ms)

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
plt.savefig('./reports/poster_ohbm/R1_pca25_mean.eps', format='eps')
plt.close()

sns.distplot(error_young, color='green', label="Young")
sns.distplot(error_old, color='blue', label="Adult")
sns.distplot(error_ms, color='orange', label="MS")
plt.legend()
plt.ylim(0,1.5)
plt.xlim(0,16)
plt.savefig('./reports/poster_ohbm/R1_pca_errordist_abs.svg', format='svg')
plt.close()

sns.distplot(error_young_lim, color='green', label="Young")
sns.distplot(error_old_lim, color='blue', label="Adult")
sns.distplot(error_ms_lim, color='orange', label="MS")
plt.legend()
plt.ylim(0,0.7)
plt.xlim(-7.5,15)
plt.savefig('./reports/poster_ohbm/R1_pca_errordist.svg', format='svg')
plt.close()

np.mean(error_young), np.mean(error_old), np.mean(error_ms), np.mean(error_young_lim), np.mean(error_old_lim), np.mean(error_ms_lim)
(3.661165680746049,
 9.697803239664681,
 11.828568007759033,
 -0.053332811366330855,
 0.27872441929038866,
 7.991847080177768)

error_young_pca_r1 = mean_pred_young - y_young
error_old_pca_r1 = mean_pred_old - y_old
error_ms_pca_r1_1 = mean_pred_ms1
error_ms_pca_r1_2 = mean_pred_ms2

"""
CT - ElasticNet 
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')
cortical_parc='volume'
dataset = 'stanford_ms_run1'
cortical_mat_ms1, age_ms1, subs, _ = load_dataset(dataset, 
                                                cortical_parc=cortical_parc,
                                                measure_type='t1')
dataset = 'stanford_ms_run2'
cortical_mat_ms2, age_ms2, subs, _ = load_dataset(dataset, 
                                                cortical_parc=cortical_parc,
                                                measure_type='t1')
cortical_mat_ms1_ct = np.delete(cortical_mat_ms1, 74, axis=0)
cortical_mat_ms2_ct = np.delete(cortical_mat_ms2, 74, axis=0)

NUM_REPS = 1000
X = data_ct.T

AGE = 25

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


#prepare a range of parameters to test
alphas = np.array([0.65]) 
l1_ratio=np.array([0.9])
#create and fit a ridge regression model, testing each alpha
model = linear_model.ElasticNet() #We have chosen to just normalize the data by default, you could GridsearchCV this is you wanted
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas, l1_ratio=l1_ratio))
grid.fit(data_old, y_old)


# we want 38 features
grid.best_estimator_.alpha=0.06
grid.best_estimator_.l1_ratio=0.999
lm = linear_model.ElasticNet(alpha=grid.best_estimator_.alpha, l1_ratio=grid.best_estimator_.l1_ratio)
error_old = np.zeros(NUM_REPS)
error_old_lim = np.zeros(NUM_REPS)
error_old_tot = np.empty((data_old.shape[0],NUM_REPS))
error_old_tot[:] = np.nan
indices = np.arange(data_old.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old, idx1, idx2 = train_test_split(data_old, y_old, indices)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    error_old[idx] = metrics.mean_absolute_error(y_test_old, y_pred)
    error_old_tot[idx2,idx] = y_pred
    error_old_lim[idx] = np.mean(y_test_old - y_pred)
mean_pred_old = np.nanmean(error_old_tot, axis=1)
np.mean(error_old)
np.count_nonzero(lm.coef_)

#plt.scatter(mean_pred_old, y_old)
#plt.plot(range(15,100),range(15,100))
#plt.xlabel("predicted age")
#plt.ylabel("real age")
cortical_mat_ms1_ct = np.delete(cortical_mat_ms1_ct, np.where(areas_notdeg2==1), axis=0) 
cortical_mat_ms2_ct = np.delete(cortical_mat_ms2_ct, np.where(areas_notdeg2==1), axis=0) 

error_ms = np.zeros(NUM_REPS)
error_ms_lim = np.zeros(NUM_REPS)
error_ms1_tot = np.empty((cortical_mat_ms1_ct.shape[1],NUM_REPS))
error_ms1_tot[:] = np.nan
error_ms2_tot = np.empty((cortical_mat_ms1_ct.shape[1],NUM_REPS))
error_ms2_tot[:] = np.nan
indices = np.arange(cortical_mat_ms1_ct.shape[1])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    y_pred_ms1 = lm.predict(cortical_mat_ms1_ct.T)
    y_pred_ms2 = lm.predict(cortical_mat_ms2_ct.T)
    error_ms[idx] = np.mean((metrics.mean_absolute_error(y_pred_ms1, age_ms1), metrics.mean_absolute_error(y_pred_ms2, age_ms2)))
    error_ms_lim[idx] = np.mean((np.mean(y_pred_ms1 - age_ms1), np.mean(y_pred_ms2 - age_ms2)))
    error_ms1_tot[:,idx] = y_pred_ms1
    error_ms2_tot[:,idx] = y_pred_ms2
mean_pred_ms1 = np.nanmean(error_ms1_tot, axis=1)
mean_pred_ms2 = np.nanmean(error_ms2_tot, axis=1)
np.mean(error_ms_lim)


# we want 44 features
# we want 43 features with no reading
grid.best_estimator_.alpha=0.06
grid.best_estimator_.l1_ratio=0.8
lm = linear_model.ElasticNet(alpha=grid.best_estimator_.alpha, l1_ratio=grid.best_estimator_.l1_ratio) 
error_young = np.zeros(NUM_REPS)
error_young_lim = np.zeros(NUM_REPS)
error_young_tot = np.empty((data_young.shape[0],NUM_REPS))
error_young_tot[:] = np.nan
indices = np.arange(data_young.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young, idx1, idx2 = train_test_split(data_young, y_young, indices)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    error_young[idx] = metrics.mean_absolute_error(y_test_young, y_pred)
    error_young_lim[idx] = np.mean(y_test_young - y_pred)
    error_young_tot[idx2,idx] = y_pred
mean_pred_young = np.nanmean(error_young_tot, axis=1)
np.mean(error_young)
np.count_nonzero(lm.coef_)


plt.scatter(mean_pred_old, y_old)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")

plt.scatter(np.hstack((mean_pred_ms1)), np.hstack((age_ms1)))
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")
plt.title("Age prediction ElasticNet80")  

plt.scatter(mean_pred_young, y_young)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")

plt.scatter(np.hstack(( mean_pred_ms2)), np.hstack((age_ms2)), c='orange', edgecolors='k')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")
plt.title("Age prediction ElasticNet80 CT") 
plt.savefig('./reports/poster_ohbm/ct_elastic_net25_mean.eps', format='eps')


sns.distplot(error_young, color='green', label="Young")
sns.distplot(error_old, color='blue', label="Adult")
sns.distplot(error_ms, color='orange', label="MS")
plt.legend()
plt.ylim(0,1.5)
plt.xlim(0,16)
plt.savefig('./reports/poster_ohbm/ct_en_errordist_abs.svg', format='svg')
plt.close()

sns.distplot(error_young_lim, color='green', label="Young")
sns.distplot(error_old_lim, color='blue', label="Adult")
sns.distplot(error_ms_lim, color='orange', label="MS")
plt.legend()
plt.ylim(0,0.7)
plt.xlim(-7.5,12.5)
plt.savefig('./reports/poster_ohbm/ct_en_errordist.svg', format='svg')
plt.close()

np.mean(error_young), np.mean(error_old), np.mean(error_ms), np.mean(error_young_lim), np.mean(error_old_lim), np.mean(error_ms_lim)
(3.4246547636922897,
 10.836275505857783,
 9.128927197234129,
 -0.09299417310705645,
 0.6022617562178534,
 3.0873736575113937)

error_young_en_ct = mean_pred_young - y_young
error_old_en_ct = mean_pred_old - y_old
error_ms_en_ct_1 = mean_pred_ms1
error_ms_en_ct_2 = mean_pred_ms2
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

lm = linear_model.LinearRegression()
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
mean_pred_ms1 = np.nanmean(error_ms1_tot, axis=1)
mean_pred_ms2 = np.nanmean(error_ms2_tot, axis=1)
np.mean(error_ms)


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

plt.scatter(np.hstack((mean_pred_ms1)), np.hstack((age_ms1)))
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")

plt.scatter(mean_pred_young, y_young)
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")


plt.scatter(np.hstack((mean_pred_ms2)), np.hstack((age_ms2)), c='orange', edgecolors='k')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")
plt.title("Age prediction PCA CT") 
plt.savefig('./reports/poster_ohbm/ct_pca_mean.eps', format='eps')

sns.distplot(error_young, color='green', label="Young")
sns.distplot(error_old, color='blue', label="Adult")
sns.distplot(error_ms, color='orange', label="MS")
plt.legend()
plt.ylim(0,1.5)
plt.xlim(0,16)
plt.savefig('./reports/poster_ohbm/ct_pca_errordist_abs.svg', format='svg')
plt.close()

sns.distplot(error_young_lim, color='green', label="Young")
sns.distplot(error_old_lim, color='blue', label="Adult")
sns.distplot(error_ms_lim, color='orange', label="MS")
plt.legend()
plt.ylim(0,0.7)
plt.xlim(-7.5,12.5)
plt.savefig('./reports/poster_ohbm/ct_pca_errordist.svg', format='svg')
plt.close()


np.mean(error_young), np.mean(error_old), np.mean(error_ms), np.mean(error_young_lim), np.mean(error_old_lim), np.mean(error_ms_lim)
(3.402629237192779,
 10.844655284593019,
 9.060298817089363,
 -0.07660424322487833,
 0.7415212760464555,
 3.143667217457892)


error_young_pca_ct = mean_pred_young - y_young
error_old_pca_ct = mean_pred_old - y_old
error_ms_pca_ct_1 = mean_pred_ms1
error_ms_pca_ct_2 = mean_pred_ms2

""""
statistics 
""""
from scipy.stats import ttest_ind 

y_young
y_old
y_ms_1 = age_ms1
y_ms_2 = age_ms2

error_young_pca_ct
error_old_pca_ct
error_ms_pca_ct

error_young_pca_r1
error_old_pca_r1
error_ms_pca_r1

error_young_en_ct
error_old_en_ct
error_ms_en_ct

error_young_en_r1
error_old_en_r1
error_ms_en_r1

#age_ms = [41, 61]
y_old
#y_old[(y_old >= 41) & (y_old <= 61)] # y_old[(y_old >= 40) & (y_old <= 62)]

idx_healthy_old_ms_1 = np.where((y_old >= 39) & (y_old <= 59))
idx_healthy_old_ms_2 = np.where((y_old >= 41) & (y_old <= 61))

# error_young_pca_ct error_young_pca_r1
t, p = ttest_ind(np.abs(error_young_pca_ct), np.abs(error_young_pca_r1))
sns.boxplot(data=[np.abs(error_young_pca_ct) , np.abs(error_young_pca_r1)], color='g')
plt.savefig('./reports/poster_ohbm/young_ct_r1_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()

# error_old_pca_ct error_old_pca_r1
t, p = ttest_ind(np.abs(error_old_pca_ct), np.abs(error_old_pca_r1))
sns.boxplot(data=[np.abs(error_old_pca_ct) , np.abs(error_old_pca_r1)], color='b')
plt.savefig('./reports/poster_ohbm/old_ct_r1_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()


# error_young_en_ct error_young_en_r1
t, p = ttest_ind(np.abs(error_young_en_ct), np.abs(error_young_en_r1))
sns.boxplot(data=[np.abs(error_young_en_ct) , np.abs(error_young_en_r1)], color='g')
plt.savefig('./reports/poster_ohbm/young_ct_r1_stats_en_' + str(p) + '.svg', format='svg')
plt.close()

# error_old_en_ct error_old_en_r1
t, p = ttest_ind(np.abs(error_old_en_ct), np.abs(error_old_en_r1))
sns.boxplot(data=[np.abs(error_old_en_ct) , np.abs(error_old_en_r1)], color='b')
plt.savefig('./reports/poster_ohbm/old_ct_r1_stats_en_' + str(p) + '.svg', format='svg')
plt.close()


"""
ms stats
"""

# error_old_pca_ct[idx_healthy_old_ms]
t, p = ttest_ind(error_old_pca_ct[idx_healthy_old_ms_1], error_ms_pca_ct_1 - y_ms_1)
sns.boxplot(data=[error_old_pca_ct[idx_healthy_old_ms_1] , error_ms_pca_ct_1 - y_ms_1], color='orange')
plt.savefig('./reports/poster_ohbm/old_ct_ms1_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()
# error_old_pca_ct[idx_healthy_old_ms]
t, p = ttest_ind(error_old_pca_ct[idx_healthy_old_ms_2], error_ms_pca_ct_2 - y_ms_2)
sns.boxplot(data=[error_old_pca_ct[idx_healthy_old_ms_2] , error_ms_pca_ct_2 - y_ms_2], color='orange')
plt.savefig('./reports/poster_ohbm/old_ct_ms2_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()

# error_old_pca_r1[idx_healthy_old_ms]
t, p = ttest_ind(error_old_pca_r1[idx_healthy_old_ms_1], error_ms_pca_r1_1 - y_ms_1) # np.delete(error_ms_pca_r1_1, [1,4]) - np.delete(y_ms_1, [1,4]) # removing outliers, still significant
sns.boxplot(data=[error_old_pca_r1[idx_healthy_old_ms_1] , error_ms_pca_r1_1 - y_ms_1], color='orange')
plt.savefig('./reports/poster_ohbm/old_r1_ms1_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()
# error_old_pca_r1[idx_healthy_old_ms]
t, p = ttest_ind(error_old_pca_r1[idx_healthy_old_ms_2], error_ms_pca_r1_2 - y_ms_2)
sns.boxplot(data=[error_old_pca_r1[idx_healthy_old_ms_2] , error_ms_pca_r1_2 - y_ms_2], color='orange')
plt.savefig('./reports/poster_ohbm/old_r1_ms2_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()

# error_old_en_ct[idx_healthy_old_ms]
t, p = ttest_ind(error_old_en_ct[idx_healthy_old_ms_1], error_ms_en_ct_1 - y_ms_1)
sns.boxplot(data=[error_old_en_ct[idx_healthy_old_ms_1] , error_ms_en_ct_1 - y_ms_1], color='orange')
plt.savefig('./reports/poster_ohbm/old_ct_ms1_stats_en_' + str(p) + '.svg', format='svg')
plt.close()
# error_old_en_ct[idx_healthy_old_ms]
t, p = ttest_ind(error_old_en_ct[idx_healthy_old_ms_2], error_ms_en_ct_2 - y_ms_2)
sns.boxplot(data=[error_old_en_ct[idx_healthy_old_ms_2] , error_ms_en_ct_2 - y_ms_2], color='orange')
plt.savefig('./reports/poster_ohbm/old_ct_ms2_stats_en_' + str(p) + '.svg', format='svg')
plt.close()

# error_old_en_r1[idx_healthy_old_ms]
t, p = ttest_ind(error_old_en_r1[idx_healthy_old_ms_1], error_ms_en_r1_1 - y_ms_1) # np.delete(error_ms_en_r1_1, [1,4]) - np.delete(y_ms_1, [1,4]) # removing outliers, still significant
sns.boxplot(data=[error_old_en_r1[idx_healthy_old_ms_1] , error_ms_en_r1_1 - y_ms_1], color='orange')
plt.savefig('./reports/poster_ohbm/old_r1_ms1_stats_en_' + str(p) + '.svg', format='svg')
plt.close()
# error_old_en_r1[idx_healthy_old_ms]
t, p = ttest_ind(error_old_en_r1[idx_healthy_old_ms_2], error_ms_en_r1_2 - y_ms_2)
sns.boxplot(data=[error_old_en_r1[idx_healthy_old_ms_2] , error_ms_en_r1_2 - y_ms_2], color='orange')
plt.savefig('./reports/poster_ohbm/old_r1_ms2_stats_en_' + str(p) + '.svg', format='svg')
plt.close()


