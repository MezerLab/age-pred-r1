#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:49:06 2020

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
PCA
"""
"""
R1
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')

dataset = 'reading'
data_matrix_reading, age_reading, subs_reading, _ = load_dataset(dataset,
                                                                 cortical_parc='midgray',
                                                                 measure_type = 'r1')
no_preterm =['s016', 's088',  's114', 's027', 's041', 
              's121', 's132', 's143', 's057', 
             's061', 's079',  's070', 's067',  
             's080', 's148', 's112',  's117', 's160', 's113', ]
#             
#             # ringing
#             's007','s034','s037','s045','s046','s064','s078','s082',
#             's083','s083_2','s092_dti30','s097_2','s101','s109','s111','s116',
#             's120','s123','s126','s126_2','s127','s129','s133','s137',
#             's141','s142','s144','s145','s146','s147','s153','s154',
#             's157','s168','s169','s171','s175',
#             's006','s038','AOK07_run1','s008_2','s040','s055',
#             's062','s081','s086','s096','s102_2','s125','s134','s138']
no_preterm_idx = [index for index, elem in enumerate(subs_reading) if elem in no_preterm]
data_reading_ctrl = data_matrix_reading[:,no_preterm_idx].copy()
y_reading_ctrl = age_reading[no_preterm_idx].copy()
subs_reading_ctrl = subs_reading[no_preterm_idx].copy()
data_reading_ctrl = np.delete(data_reading_ctrl, np.where(areas_notdeg2==1), axis=1) 
#'s016', 's027', 's061', 's067', 's113', 's114', 's117', 's121',
# 's143', 's148'.
# gender = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0]


NUM_REPS = 1000
X = data_t1.T


adults = np.where(y_t1>AGE)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X, adults, axis=0) 
data_young = np.delete(data_young, np.where(areas_notdeg2==1), axis=1) 
#df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report(X=data_young, features_=np.delete(areas, np.where(areas_notdeg2==1)), save_plot=False, fig_dpi=50)



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



X_ms1 = data_reading_ctrl.T
X_ms1 = np.delete(X_ms1, np.where(areas_notdeg2==1), axis=1) 

error_ms = np.zeros(NUM_REPS)
error_ms_lim = np.zeros(NUM_REPS)
error_ms1_tot = np.empty((10, NUM_REPS))
error_ms1_tot[:] = np.nan
indices = np.arange(data_reading_ctrl.shape[0])
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_young, y_young)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca_ms1 = pca.transform(X_ms1)
    lm.fit(X_train_pca, y_train)
    y_pred_ms1 = lm.predict(X_test_pca_ms1)
    error_ms[idx] = np.mean((metrics.mean_absolute_error(y_pred_ms1, y_reading_ctrl)))
    error_ms_lim[idx] = np.mean((np.mean(y_pred_ms1 - y_reading_ctrl)))
    error_ms1_tot[:,idx] = y_pred_ms1
mean_pred_ms1 = np.nanmean(error_ms1_tot, axis=1)
np.mean(error_ms)



plt.scatter(mean_pred_young, y_young, c='green')
plt.xlabel("predicted age")
plt.ylabel("real age")

plt.scatter(np.hstack(( mean_pred_ms1)), np.hstack(( y_reading_ctrl)), c='green', edgecolors='k')
plt.plot(range(5,30),range(5,30))
plt.xlabel("predicted age")
plt.ylabel("real age ms")
plt.xlim(0,35)
plt.ylim(0,30)
plt.title("Age prediction PCA R1") 
plt.savefig('./reports/poster_ohbm/R1_pcakatie_mean.eps', format='eps')
plt.close()


error_young_pca_r1 = mean_pred_young - y_young
error_old_pca_r1 = mean_pred_old - y_old
error_ms_pca_r1 = np.hstack((mean_pred_ms1))


""""
statistics 
""""
from scipy.stats import ttest_ind 


y_ms = y_reading_ctrl


#age_ms = [41, 61]
y_old
#y_old[(y_old >= 41) & (y_old <= 61)] # y_old[(y_old >= 40) & (y_old <= 62)]

idx_healthy_old_ms = np.where((y_young >= 6) & (y_young <= 8.7))


"""
ms stats
"""


# error_old_pca_r1[idx_healthy_old_ms]
t, p = ttest_ind(error_young_pca_r1[idx_healthy_old_ms], error_ms_pca_r1 - y_ms)
sns.boxplot(data=[error_young_pca_r1[idx_healthy_old_ms] , error_ms_pca_r1 - y_ms], color='orange')
plt.savefig('./reports/poster_ohbm/young_r1_katie_stats_pca_' + str(p) + '.svg', format='svg')
plt.close()
