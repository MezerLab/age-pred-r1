#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:48:20 2020

@author: asier.erramuzpe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:44:54 2020

@author: asier.erramuzpe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:17:17 2020

@author: asier.erramuzpe
"""



import numpy as np
from src.data.make_dataset import (load_cort,
                                   load_dataset,
                                   load_cort_areas,
                                   )
from sklearn import decomposition
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
R2
"""

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
pca = decomposition.FastICA(n_components=NUM_PCS)

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
NUM_PCS = 27 # pca_idx
pca = decomposition.FastICA(n_components=NUM_PCS)
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
plt.title("Age prediction FastICA R1") 
plt.savefig('./reports/figures/rebuttal/lle/FastICA_pca25_mean.eps', format='eps')
plt.close()

np.mean(error_young), np.mean(error_old)
#(3.9096989929605614, 9.498516180098715)

error_young_pca_r1_lle = mean_pred_young - y_young
error_old_pca_r1_lle = mean_pred_old - y_old


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
pca = decomposition.FastICA(n_components=NUM_PCS)

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


NUM_PCS = 43
pca = decomposition.FastICA(n_components=NUM_PCS)
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

plt.scatter(mean_pred_young, y_young, c='green')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age")
plt.xlim(0,100)
plt.title("Age prediction FastICA CT") 
plt.savefig('./reports/figures/rebuttal/lle/FastICA_pca25_mean_ct.eps', format='eps')
plt.close()

np.mean(error_young), np.mean(error_old)
#(3.6129334902860504, 10.649367162215844)

error_young_pca_ct_lle = mean_pred_young - y_young
error_old_pca_ct_lle = mean_pred_old - y_old

""""
statistics 
""""
from scipy.stats import ttest_ind 

# error_young_pca_ct error_young_pca_r1
t, p = ttest_ind(np.abs(error_young_pca_ct_lle), np.abs(error_young_pca_r1_lle))
sns.boxplot(data=[np.abs(error_young_pca_ct_lle) , np.abs(error_young_pca_r1_lle)], color='g')
plt.savefig('./reports/figures/rebuttal/lle/young_ct_r1_stats_FastICA_' + str(p) + '.svg', format='svg')
plt.close()

# error_old_pca_ct error_old_pca_r1
t, p = ttest_ind(np.abs(error_old_pca_ct_lle), np.abs(error_old_pca_r1_lle))
sns.boxplot(data=[np.abs(error_old_pca_ct_lle) , np.abs(error_old_pca_r1_lle)], color='b')
plt.savefig('./reports/figures/rebuttal/lle/old_ct_r1_stats_FastICA_' + str(p) + '.svg', format='svg')
plt.close()





# pca vs fastICA r1
t, p = ttest_ind(np.abs(error_young_pca_r1), np.abs(error_young_pca_r1_lle))
sns.boxplot(data=[np.abs(error_young_pca_r1) , np.abs(error_young_pca_r1_lle)], color='g')
plt.savefig('./reports/figures/rebuttal/lle/pca_vs_FastICA_r1_young_' + str(p) + '.svg', format='svg')
plt.close()

# pca vs fastICA r1
t, p = ttest_ind(np.abs(error_old_pca_r1), np.abs(error_old_pca_r1_lle))
sns.boxplot(data=[np.abs(error_old_pca_r1) , np.abs(error_old_pca_r1_lle)], color='b')
plt.savefig('./reports/figures/rebuttal/lle/pca_vs_FastICA_r1_old_' + str(p) + '.svg', format='svg')
plt.close()

# pca vs fastICA ct
t, p = ttest_ind(np.abs(error_young_pca_ct), np.abs(error_young_pca_ct_lle))
sns.boxplot(data=[np.abs(error_young_pca_ct) , np.abs(error_young_pca_ct_lle)], color='g')
plt.savefig('./reports/figures/rebuttal/lle/pca_vs_FastICA_ct_young_' + str(p) + '.svg', format='svg')
plt.close()

# pca vs fastICA ct
t, p = ttest_ind(np.abs(error_old_pca_ct), np.abs(error_old_pca_ct_lle))
sns.boxplot(data=[np.abs(error_old_pca_ct) , np.abs(error_old_pca_ct_lle)], color='b')
plt.savefig('./reports/figures/rebuttal/lle/pca_vs_FastICA_ct_old_' + str(p) + '.svg', format='svg')
plt.close()
