#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:14:18 2020

@author: asier.erramuzpe
"""

import numpy as np
from src.data.make_dataset import (load_cort,
                                   load_dataset,
                                   load_cort_areas,
                                   )
from src.visualization.visualize import (plot_areas,
                                         plot_brain_surf,
                                         )
areas = load_cort_areas()
NUM_AREAS = 148
areas_notdeg2_idx = [  0,   1,   4,   5,  11,  12,  13,  17,  20,  21,  22,  23,  30,  31,  33,
                       34,  36,  38,  41,  42,  46,  47,  49,  61,  62,  63,  69, 71, 74,
                       75,  79,  94,  95,  96,  97, 104, 105, 108, 115, 116, 120, 136, 137,
                       143] 
areas_notdeg2 = np.zeros(NUM_AREAS)
areas_notdeg2[areas_notdeg2_idx] = 1
N_FEAT = 40

"""
CT
"""
data_t1, y_t1, subs_t1 = load_cort(measure='t1', cort='volume')
dataset = 'stanford_ms_run1'
cortical_mat_ms1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='volume',
                                                  measure_type='t1')
dataset = 'stanford_ms_run2'
cortical_mat_ms2, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='volume',
                                                  measure_type='t1')
cortical_mat_ms1 = np.delete(cortical_mat_ms1, 74, axis=0)
cortical_mat_ms2 = np.delete(cortical_mat_ms2, 74, axis=0)
idx = np.where((y_t1 > 40) & (y_t1 < 62))
mean_vals_t1 = np.mean(data_t1[:, idx[0]], axis=1)
mean_vals_ms = np.mean(np.hstack((cortical_mat_ms1, cortical_mat_ms2)), axis=1)
diff_ct = mean_vals_t1 - mean_vals_ms # contrarily to T1, here it's CT healthy supposed to be higher
diff_ct[np.where(areas_notdeg2==1)] = 0
"""
most loss areas CT
"""
idx_maxloss_ct = diff_ct.argsort()[-N_FEAT:][::-1] # ordered from bottom to top (is better to have higher CT)
diff_ct[idx_maxloss_ct[0]]
"""
R1
"""
data_t1, y_t1, subs_t1 = load_cort(measure='r1', cort='midgray')
dataset = 'stanford_ms_run1'
cortical_mat_ms1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                  measure_type='r1_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                  measure_type='r1_les')
idx = np.where((y_t1 > 40) & (y_t1 < 62))
mean_vals_t1 = np.mean(data_t1[:, idx[0]], axis=1)
mean_vals_ms = np.mean(np.hstack((cortical_mat_ms1, cortical_mat_ms2)), axis=1)
diff_r1 = mean_vals_t1 - mean_vals_ms
diff_r1[np.where(areas_notdeg2==1)] = 0
"""
most loss areas R1
"""
idx_maxloss_r1 = diff_r1.argsort()[-N_FEAT:][::-1] # ordered from bottom to top (is better to have higher CT)
diff_r1[idx_maxloss_r1[0]]


"""
Positive and negative weighted areas
"""
N_FEAT_WEIGHT = 10
#mean_pred_old_r1 = mean_pred_old
#mean_pred_old_r1_pca = np.mean(PCA_influence_old, axis=1)
for idx in areas_notdeg2_idx:
    mean_pred_old_r1 = np.insert(mean_pred_old_r1, idx, 0)
#mean_pred_old_ct
for idx in areas_notdeg2_idx:
    mean_pred_old_ct = np.insert(mean_pred_old_ct, idx, 0)
    
idx_pos_weight_r1 = mean_pred_old_r1.argsort()[-N_FEAT_WEIGHT:][::-1] 
idx_neg_weight_r1 = mean_pred_old_r1.argsort()[:N_FEAT_WEIGHT] 
#mean_pred_old[mean_pred_old.argsort()[-N_FEAT:][::-1]]

idx_pos_weight_ct = mean_pred_old_ct.argsort()[-N_FEAT_WEIGHT:][::-1] 
idx_neg_weight_ct = mean_pred_old_ct.argsort()[:N_FEAT_WEIGHT] 
#mean_pred_old_ct[mean_pred_old_ct.argsort()[-N_FEAT:][::-1]]

"""
Now, the prediction
"""
"""
1.- Remove important weights areas from R1 prediction
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
areas_notdeg2[idx_maxloss_r1] = 1


#val_mat = np.zeros(areas.shape)
#val_mat[idx_pos_weight_r1] = 1
#val_mat[np.where(val_mat==0)] = np.nan
#make_freeview_fig(val_mat, 'neg_weights_r1')


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

NUM_PCS = 28 # pca_idx
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
plt.savefig('./reports/poster_ohbm/R1_pca25_mean_removing_neg_weights_areas.eps', format='eps')
plt.close()

sns.distplot(error_young, color='green', label="Young")
sns.distplot(error_old, color='blue', label="Adult")
sns.distplot(error_ms, color='orange', label="MS")
plt.legend()
plt.ylim(0,1.5)
plt.xlim(0,16)
plt.savefig('./reports/poster_ohbm/R1_pca_errordist_abs_removing_neg_weights_areas.svg', format='svg')
plt.close()

sns.distplot(error_young_lim, color='green', label="Young")
sns.distplot(error_old_lim, color='blue', label="Adult")
sns.distplot(error_ms_lim, color='orange', label="MS")
plt.legend()
plt.ylim(0,0.7)
plt.xlim(-7.5,12.5)
plt.savefig('./reports/poster_ohbm/R1_pca_errordist_removing_neg_weights_areas.svg', format='svg')
plt.close()

np.mean(error_young), np.mean(error_old), np.mean(error_ms), np.mean(error_young_lim), np.mean(error_old_lim), np.mean(error_ms_lim)
#regular 
(3.6497007538261994,
 9.624854160335465,
 12.074310741250283,
 -0.0009330466928607102,
 0.01704305169407427,
 8.43018249180498)
# removing_pos_weights_areas = MS not older anymore
(3.4795694993261503,
 11.547045022065593,
 10.96039434502208,
 -0.08742705478167098,
 0.7074337178285648,
 5.233439463689141)
# removing_neg_weights_areas = MS prediction still older, error of prediction increases
(3.719585675739313,
 11.618942664766356,
 12.44713291022428,
 -0.012259037651033908,
 0.35492828888209205,
 8.561471192963735)





idx_pos_weight_ct
idx_neg_weight_ct 

idx_maxloss_ct

"""
2.- Add important difference areas to CT prediction
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
areas_notdeg2 = np.ones(NUM_AREAS)
areas_notdeg2[areas_notdeg2_idx] = 1
#areas_notdeg2[idx_pos_weight_ct] = 1
#areas_notdeg2[idx_neg_weight_ct] = 1
areas_notdeg2[idx_maxloss_ct] = 0


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
NUM_PCS = 19
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

NUM_PCS = 19
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


plt.scatter(np.hstack(( mean_pred_ms2)), np.hstack(( age_ms2)), c='orange', edgecolors='k')
plt.plot(range(15,100),range(15,100))
plt.xlabel("predicted age")
plt.ylabel("real age ms")
plt.xlim(0,100)
plt.title("Age prediction PCA R1") 
plt.savefig('./reports/poster_ohbm/CT_pca25_mean_only_MS_affected_areas.eps', format='eps')
plt.close()

sns.distplot(error_young, color='green', label="Young")
sns.distplot(error_old, color='blue', label="Adult")
sns.distplot(error_ms, color='orange', label="MS")
plt.legend()
plt.ylim(0,1.5)
plt.xlim(0,16)
plt.savefig('./reports/poster_ohbm/CT_pca_errordist_abs_only_MS_affected_areas.svg', format='svg')
plt.close()

sns.distplot(error_young_lim, color='green', label="Young")
sns.distplot(error_old_lim, color='blue', label="Adult")
sns.distplot(error_ms_lim, color='orange', label="MS")
plt.legend()
plt.ylim(0,0.7)
plt.xlim(-10,15)
plt.savefig('./reports/poster_ohbm/CT_pca_errordist_only_MS_affected_areas.svg', format='svg')
plt.close()

np.mean(error_young), np.mean(error_old), np.mean(error_ms), np.mean(error_young_lim), np.mean(error_old_lim), np.mean(error_ms_lim)
#regular 
(3.409257602069457,
 10.455091965471631,
 8.370891409370154,
 -0.09147776726237428,
 0.45756274522191415,
 2.8972788742611337)
# only using areas that differ from MS
(4.11758296007939,
 11.675979947240746,
 10.780486296190652,
 -0.002836893974971053,
 0.24078898365613724,
 7.617310760916809)



"""
ms stats
"""
""""
statistics 
""""
from scipy.stats import ttest_ind 
error_young_pca_ct = mean_pred_young - y_young
error_old_pca_ct = mean_pred_old - y_old
error_ms_pca_ct_1 = mean_pred_ms1
error_ms_pca_ct_2 = mean_pred_ms2
y_ms_1 = age_ms1
y_ms_2 = age_ms2

idx_healthy_old_ms_1 = np.where((y_old >= 39) & (y_old <= 59))
idx_healthy_old_ms_2 = np.where((y_old >= 41) & (y_old <= 61))

# error_old_pca_r1[idx_healthy_old_ms]
t, p = ttest_ind(error_old_pca_ct[idx_healthy_old_ms_1], error_ms_pca_ct_1 - y_ms_1)
sns.boxplot(data=[error_old_pca_ct[idx_healthy_old_ms_1] , error_ms_pca_ct_1 - y_ms_1], color='orange')
plt.savefig('./reports/poster_ohbm/old_ct_ms1_stats_pca_supmat_' + str(p) + '.svg', format='svg')
plt.close()
# error_old_pca_r1[idx_healthy_old_ms]
t, p = ttest_ind(error_old_pca_ct[idx_healthy_old_ms_2], error_ms_pca_ct_2 - y_ms_2)
sns.boxplot(data=[error_old_pca_ct[idx_healthy_old_ms_2] , error_ms_pca_ct_2 - y_ms_2], color='orange')
plt.savefig('./reports/poster_ohbm/old_ct_ms2_stats_pca_supmat_' + str(p) + '.svg', format='svg')
plt.close()
