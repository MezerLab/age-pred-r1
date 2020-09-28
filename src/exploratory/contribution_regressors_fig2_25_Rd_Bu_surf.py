#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:31:08 2019

@author: asier.erramuzpe
"""

import numpy as np
from src.data.make_dataset import (load_cort,
                                   load_dataset,
                                   load_cort_areas,
                                   )
from src.visualization.visualize import (plot_areas,
                                         plot_area_values,
                                         plot_brain_surf,
                                         make_freeview_fig,
                                         )
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import (GridSearchCV,
                                     train_test_split,
                                     )
import seaborn as sns
import matplotlib.pyplot as plt

NUM_AREAS = 148
areas_notdeg2_idx = [  0,   1,   4,   5,  11,  12,  13,  17,  20,  21,  22,  23,  30,  31,  33,
                       34,  36,  38,  41,  42,  46,  47,  49,  61,  62,  63,  69, 71, 74,
                       75,  79,  94,  95,  96,  97, 104, 105, 108, 115, 116, 120, 136, 137,
                       143] 
areas_notdeg2 = np.zeros(NUM_AREAS)
areas_notdeg2[areas_notdeg2_idx] = 1

areas = load_cort_areas()
areas = np.delete(areas, np.where(areas_notdeg2==1))

def calc_influence_PCA(X_train, lm, NUM_PCS):
    
    df_pca, X_pca, pca_evr, df_corr, df_n, df_na, df_nabv, df_rank = pca_full_report_fast(X=X_train, features_=areas, save_plot=False)
    
    return np.dot(df_n.iloc[:, :NUM_PCS], lm.coef_)

def compute_precentile_matrix(a, percentile=75):
    
    a_pos_idx = np.where(a>0)
    a_pos = np.zeros(a.shape)
    a_pos[a_pos_idx] = np.abs(a[a_pos_idx])
    perc = np.percentile(a_pos[a_pos_idx], percentile)
    a_neg_idx = np.where(a<0)
    a_neg = np.zeros(a.shape)
    a_neg[a_neg_idx] = np.abs(a[a_neg_idx])
    a_pos_idx_perc = np.where(a_pos>perc)
    a_pos_perc = np.zeros(a.shape)
    perc = np.percentile(a_neg[a_neg_idx], percentile)
    a_neg_idx_perc = np.where(a_neg>perc)
    a_pos_perc[a_neg_idx_perc] = -a_neg[a_neg_idx_perc]
    a_pos_perc[a_pos_idx_perc] = np.abs(a_pos[a_pos_idx_perc])

    return a_pos_perc

AGE = 25
percentile = 75

"""
R1 - ElasticNet
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')

NUM_REPS = 100
X = data_t1.T

youngsters = np.where(y_t1<=AGE)
y_old = np.delete(y_t1, youngsters)
data_old = np.delete(X, youngsters, axis=0) 
data_old = np.delete(data_old, np.where(areas_notdeg2==1), axis=1) 

adults = np.where(y_t1>AGE)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X, adults, axis=0) 
data_young = np.delete(data_young, np.where(areas_notdeg2==1), axis=1) 

#prepare a range of parameters to test
alphas = np.array([1,]) 
l1_ratio=np.array([0.9])
model = linear_model.ElasticNet() #We have chosen to just normalize the data by default, you could GridsearchCV this is you wanted
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas, l1_ratio=l1_ratio))
grid.fit(data_old, y_old)

# we want 30 features
grid.best_estimator_.alpha=0.01
grid.best_estimator_.l1_ratio=.9999
lm = linear_model.ElasticNet(alpha=grid.best_estimator_.alpha, l1_ratio=grid.best_estimator_.l1_ratio)
error_old_tot = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    error_old_tot[:,idx] = lm.coef_
mean_pred_old = np.nanmean(error_old_tot, axis=1)


# we want 28 features
grid.best_estimator_.alpha=0.0055
grid.best_estimator_.l1_ratio=0.999
lm = linear_model.ElasticNet(alpha=grid.best_estimator_.alpha, l1_ratio=grid.best_estimator_.l1_ratio)
error_young_tot = np.empty((data_young.shape[1], NUM_REPS))
error_young_tot[:] = np.nan
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young = train_test_split(data_young, y_young)
    lm.fit(X_train, y_train)
    error_young_tot[:,idx] = lm.coef_
mean_pred_young = np.nanmean(error_young_tot, axis=1)

"""
percentile positive old
"""

val_mat = compute_precentile_matrix(mean_pred_old, percentile=percentile)
val_mat[np.where(val_mat==0)] = np.nan
for idx in areas_notdeg2_idx:
    val_mat = np.insert(val_mat, idx, np.nan)
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/EN_adult_t1_perc_'+str(percentile)+'.png', inflate=False, title='EN_adult_t1_perc'+str(percentile))
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/EN_adult_t1_perc_inf_'+str(percentile)+'.png', inflate=True, title='EN_adult_t1_perc'+str(percentile))
make_freeview_fig(val_mat, 'EN_adult_R1_perc_'+str(percentile)+'_fs.png')

"""
percentile positive young
"""
val_mat = compute_precentile_matrix(mean_pred_young, percentile=percentile)
val_mat[np.where(val_mat==0)] = np.nan
for idx in areas_notdeg2_idx:
    val_mat = np.insert(val_mat, idx, np.nan)
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/EN_young_t1_perc_'+str(percentile)+'.png', inflate=False, title='EN_young_t1_perc'+str(percentile))
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/EN_young_t1_perc_inf_'+str(percentile)+'.png', inflate=True, title='EN_young_t1_perc'+str(percentile))
make_freeview_fig(val_mat, 'EN_young_R1_perc_'+str(percentile)+'_fs.png')


"""
R1
"""
"""
PCA
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')


NUM_REPS = 1000
NUM_PCS = 30 
pca = PCA(n_components=NUM_PCS)

X = data_t1.T

youngsters = np.where(y_t1<=AGE)
y_old = np.delete(y_t1, youngsters)
data_old = np.delete(X, youngsters, axis=0) 
data_old = np.delete(data_old, np.where(areas_notdeg2==1), axis=1) 

adults = np.where(y_t1>AGE)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X, adults, axis=0) 
data_young = np.delete(data_young, np.where(areas_notdeg2==1), axis=1) 

lm = linear_model.LinearRegression()
PCA_influence_old = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lm.fit(X_train_pca, y_train)
    PCA_influence_old[:, idx] = calc_influence_PCA(X_train, lm, NUM_PCS)       

NUM_PCS = 28 
pca = PCA(n_components=NUM_PCS)
PCA_influence_young = np.empty((data_young.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young = train_test_split(data_young, y_young)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lm.fit(X_train_pca, y_train)
    PCA_influence_young[:, idx] = calc_influence_PCA(X_train, lm, NUM_PCS)       


"""
percentile positive old
"""
val_mat = compute_precentile_matrix(np.mean(PCA_influence_old, axis=1), percentile=percentile)
val_mat[np.where(val_mat==0)] = np.nan
for idx in areas_notdeg2_idx:
    val_mat = np.insert(val_mat, idx, np.nan)
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/PCA_adult_t1_perc_'+str(percentile)+'.png', inflate=False, title='PCA_adult_t1_perc'+str(percentile))
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/PCA_adult_t1_perc_inf_'+str(percentile)+'.png', inflate=True, title='PCA_adult_t1_perc'+str(percentile))
make_freeview_fig(val_mat, 'PCA_adult_R1_perc_'+str(percentile)+'_fs.png')



"""
percentile positive young
"""
val_mat = compute_precentile_matrix(np.mean(PCA_influence_young, axis=1), percentile=percentile)
val_mat[np.where(val_mat==0)] = np.nan
for idx in areas_notdeg2_idx:
    val_mat = np.insert(val_mat, idx, np.nan)
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/PCA_young_t1_perc_'+str(percentile)+'.png', inflate=False, title='PCA_young_t1_perc'+str(percentile))
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/PCA_young_t1_perc_inf_'+str(percentile)+'.png', inflate=True, title='PCA_young_t1_perc'+str(percentile))
make_freeview_fig(val_mat, 'PCA_young_R1_perc_'+str(percentile)+'_fs.png')


## Correlation between weights of 2 methods R1
en_young = mean_pred_young
en_young[np.where((en_young<0.1) & (en_young>-0.1))] = np.nan
pca_young = np.mean(PCA_influence_young, axis=1)
bad = ~np.logical_or(np.isnan(en_young), np.isnan(pca_young))
r, pval = pearsonr(np.compress(bad, en_young), np.compress(bad, pca_young) )
print(r, pval)
sns.regplot(x=np.compress(bad, en_young), y=np.compress(bad, pca_young), order=1, color='blue')
plt.savefig('./reports/figures/exploratory/young_R1_pca_vs_en_weights_r0.75.svg', format='svg')
plt.close()

en_old = mean_pred_old
en_old[np.where((en_old<0.1) & (en_old>-0.1))] = np.nan
pca_old = np.mean(PCA_influence_old, axis=1)
bad = ~np.logical_or(np.isnan(en_old), np.isnan(pca_old))
r, pval = pearsonr(np.compress(bad, en_old), np.compress(bad, pca_old) )
print(r, pval)
sns.regplot(x=np.compress(bad, en_old), y=np.compress(bad, pca_old), order=1, color='blue')
plt.savefig('./reports/figures/exploratory/old_R1_pca_vs_en_weights_r0.84.svg', format='svg')
plt.close()

"""
CT - ElasticNet
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')

NUM_REPS = 1000
X = data_ct.T

youngsters = np.where(y_ct<=AGE)
y_old = np.delete(y_ct, youngsters)
data_old = np.delete(X, youngsters, axis=0) 
data_old = np.delete(data_old, np.where(areas_notdeg2==1), axis=1) 

adults = np.where(y_ct>AGE)
y_young = np.delete(y_ct, adults)
data_young = np.delete(X, adults, axis=0) 
data_young = np.delete(data_young, np.where(areas_notdeg2==1), axis=1) 

model = linear_model.ElasticNet()
alphas = np.array([1,]) 
l1_ratio=np.array([0.9])
grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas, l1_ratio=l1_ratio))
grid.fit(data_old, y_old)

# we want 38 features
grid.best_estimator_.alpha=0.06
grid.best_estimator_.l1_ratio=0.999
lm = linear_model.ElasticNet(alpha=grid.best_estimator_.alpha, l1_ratio=grid.best_estimator_.l1_ratio)
error_old_tot = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    error_old_tot[:,idx] = lm.coef_
mean_pred_old_ct = np.nanmean(error_old_tot, axis=1)
#np.count_nonzero(lm.coef_)

# we want 44 features
grid.best_estimator_.alpha=0.06
grid.best_estimator_.l1_ratio=0.8
lm = linear_model.ElasticNet(alpha=grid.best_estimator_.alpha, l1_ratio=grid.best_estimator_.l1_ratio) ## to have the same amount of features (~20)
error_young_tot = np.empty((data_young.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young  = train_test_split(data_young, y_young)
    lm.fit(X_train, y_train)
    error_young_tot[:,idx] = lm.coef_
mean_pred_young_ct = np.nanmean(error_young_tot, axis=1)


"""
percentile positive old
"""
val_mat = compute_precentile_matrix(mean_pred_old_ct, percentile=percentile)
val_mat[np.where(val_mat==0)] = np.nan
for idx in areas_notdeg2_idx:
    val_mat = np.insert(val_mat, idx, np.nan)
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/EN_adult_ct_perc_'+str(percentile)+'.png', inflate=False, title='EN_adult_ct_perc'+str(percentile))
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/EN_adult_ct_perc_inf_'+str(percentile)+'.png', inflate=True, title='EN_adult_ct_perc'+str(percentile))
make_freeview_fig(val_mat, 'EN_adult_ct_perc_'+str(percentile)+'_fs.png')

"""
percentile positive young
"""
val_mat = compute_precentile_matrix(mean_pred_young_ct, percentile=percentile)
val_mat[np.where(val_mat==0)] = np.nan
for idx in areas_notdeg2_idx:
    val_mat = np.insert(val_mat, idx, np.nan)
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/EN_young_ct_perc_'+str(percentile)+'.png', inflate=False, title='EN_young_ct_perc'+str(percentile))
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/EN_young_ct_perc_inf_'+str(percentile)+'.png', inflate=True, title='EN_young_ct_perc'+str(percentile))
make_freeview_fig(val_mat, 'EN_young_ct_perc_'+str(percentile)+'_fs.png')


"""
PCA
"""
"""
CT
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')

NUM_REPS = 1000
NUM_PCS = 38
pca = PCA(n_components=NUM_PCS)

X = data_ct.T

youngsters = np.where(y_ct<=AGE)
y_old = np.delete(y_ct, youngsters)
data_old = np.delete(X, youngsters, axis=0) 
data_old = np.delete(data_old, np.where(areas_notdeg2==1), axis=1) 

adults = np.where(y_ct>AGE)
y_young = np.delete(y_ct, adults)
data_young = np.delete(X, adults, axis=0) 
data_young = np.delete(data_young, np.where(areas_notdeg2==1), axis=1) 


lm = linear_model.LinearRegression()
PCA_influence_old_ct = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lm.fit(X_train_pca, y_train)
    PCA_influence_old_ct[:, idx] = calc_influence_PCA(X_train, lm, NUM_PCS)       

NUM_PCS = 43
pca = PCA(n_components=NUM_PCS)
PCA_influence_young_ct = np.empty((data_young.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young = train_test_split(data_young, y_young)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    lm.fit(X_train_pca, y_train)
    PCA_influence_young_ct[:, idx] = calc_influence_PCA(X_train, lm, NUM_PCS)       



"""
percentile positive old
"""
val_mat = compute_precentile_matrix(np.mean(PCA_influence_old_ct, axis=1), percentile=percentile)
val_mat[np.where(val_mat==0)] = np.nan
for idx in areas_notdeg2_idx:
    val_mat = np.insert(val_mat, idx, np.nan)
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/PCA_adult_ct_perc_'+str(percentile)+'.png', inflate=False, title='PCA_adult_ct_perc'+str(percentile))
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/PCA_adult_ct_perc_inf_'+str(percentile)+'.png', inflate=True, title='PCA_adult_ct_perc'+str(percentile))
make_freeview_fig(val_mat, 'PCA_adult_ct_perc_'+str(percentile)+'_fs.png')




"""
percentile positive young
"""
val_mat = compute_precentile_matrix(np.mean(PCA_influence_young_ct, axis=1), percentile=percentile)
val_mat[np.where(val_mat==0)] = np.nan
for idx in areas_notdeg2_idx:
    val_mat = np.insert(val_mat, idx, np.nan)
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/PCA_young_ct_perc_'+str(percentile)+'.png', inflate=False, title='PCA_young_ct_perc'+str(percentile))
#plot_brain_surf(val_mat, output_file='/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/blue_red/PCA_young_ct_perc_inf_'+str(percentile)+'.png', inflate=True, title='PCA_young_ct_perc'+str(percentile))
make_freeview_fig(val_mat, 'PCA_young_ct_perc_'+str(percentile)+'_fs.png')

## Correlation between weights of 2 methods
en_young = mean_pred_young_ct
en_young[np.where((en_young<0.1) & (en_young>-0.1))] = np.nan
pca_young = np.mean(PCA_influence_young_ct, axis=1)
bad = ~np.logical_or(np.isnan(en_young), np.isnan(pca_young))
r, pval = pearsonr(np.compress(bad, en_young), np.compress(bad, pca_young) )
print(r, pval)
sns.regplot(x=np.compress(bad, en_young), y=np.compress(bad, pca_young), order=1, color='blue')
plt.savefig('./reports/figures/exploratory/young_ct_pca_vs_en_weights_r0.89.svg', format='svg')
plt.close()

en_old = mean_pred_old_ct
en_old[np.where((en_old<0.1) & (en_old>-0.1))] = np.nan
pca_old = np.mean(PCA_influence_old_ct, axis=1)
bad = ~np.logical_or(np.isnan(en_old), np.isnan(pca_old))
r, pval = pearsonr(np.compress(bad, en_old), np.compress(bad, pca_old) )
print(r, pval)
sns.regplot(x=np.compress(bad, en_old), y=np.compress(bad, pca_old), order=1, color='blue')
plt.savefig('./reports/figures/exploratory/old_ct_pca_vs_en_weights_r084.svg', format='svg')
plt.close()


 
"""
DISTRIBUTION OF THE DATASET FIGURE
"""
sns.distplot(y_t1, bins=40, kde=False)
plt.title("Distribution of the dataset")
plt.savefig('/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/ESMRMB/dist_t1.eps', format='eps', dpi=1000)


"""
EXAMPLE OF AREA
"""
AREA = 129

p = np.polyfit(y_t1, data_t1[AREA, :], deg=2)
p = np.poly1d(p)
xp = np.linspace(8, 80, 72)
plt.plot(xp, p(xp))
plt.scatter(y_t1, data_t1[AREA, :])
plt.xlabel('Age of subject')
plt.ylabel('qMRI value')
plt.title((areas[AREA], str(AREA)))
plt.savefig('/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/ESMRMB/area_t1.eps', format='eps', dpi=1000)


p = np.polyfit(y_ct, data_ct[AREA, :], deg=1)
p = np.poly1d(p)
xp = np.linspace(8, 80, 72)
plt.plot(xp, p(xp))
plt.scatter(y_ct, data_ct[AREA, :])
plt.xlabel('Age of subject')
plt.ylabel('qMRI value')
plt.title((areas[AREA], str(AREA)))
plt.savefig('/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/ESMRMB/area_ct.eps', format='eps', dpi=1000)
