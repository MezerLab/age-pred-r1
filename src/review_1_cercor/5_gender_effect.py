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
                                         plot_area_values_vmin,
                                         )
areas = load_cort_areas()
AGE = 25


# male = 1
gender = [1,0,1,1,1,1,
          0,1,0,0,0,0,
          0,1,1,0,1,0,
          1,0,1,1,0,0,
          0,0,1,1,0,0,
          0,0,0,0,1,1,
          1,1,1,0,0,0,
          1,0,0,0,1,0,
          0,0,1,0,1,1,
          1,0,0,0,1,1,
          1,0,0,1, 0,
          1,0,0,1,0,
          0,1,0,1,0,
          1,1,1,0,0,    
          0,0,0,1,0,
          1,1,1,1,1,
          1,1,0,0,1,
          0,1,0,0,0,
          0,0,0,1,1,
          1,1,1,1,0,
          1,0,0,0,0,
          1,0,0,0,0,

          0, 0, 0, 1, 0, 0, 0, 0,
          1, 0, 0, 0, 0, 0, 1, 0,
          1, 0, 0, 0, 0, 1, 0, 1,
          1, 0, 0, 0, 0, 1, 1, 0,
          1, 1, 0, 0, 1, 1, 0, 0]

#np.sum(male_gender)
#np.sum(gender_young)
#np.sum(gender_old)
male_gender = np.array(gender)[:, np.newaxis].T
female_gender = male_gender^(male_gender&1==male_gender)



"""
R1 - Linear without gender
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')

dataset = 'stanford_ms_run1'
cortical_mat_ms1_t1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='t1_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2_t1, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='t1_les')

NUM_REPS = 100
X = data_t1.T

youngsters = np.where(y_t1<AGE)
y_old = np.delete(y_t1, youngsters)
data_old = np.delete(X, youngsters, axis=0) 

adults = np.where(y_t1>AGE)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X, adults, axis=0) 

lm = linear_model.LinearRegression()
error_old_tot = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    error_old_tot[:,idx] = lm.coef_
mean_pred_old = np.nanmean(error_old_tot, axis=1)


lm = linear_model.LinearRegression()
error_young_tot = np.empty((data_young.shape[1], NUM_REPS))
error_young_tot[:] = np.nan
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young = train_test_split(data_young, y_young)
    lm.fit(X_train, y_train)
    error_young_tot[:,idx] = lm.coef_
mean_pred_young = np.nanmean(error_young_tot, axis=1)



"""
R1 - Linear with gender_male
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')
data_t1 = np.concatenate((data_t1, male_gender))

dataset = 'stanford_ms_run1'
cortical_mat_ms1_t1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='t1_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2_t1, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='t1_les')

NUM_REPS = 100
X = data_t1.T

youngsters = np.where(y_t1<AGE)
y_old = np.delete(y_t1, youngsters)
data_old = np.delete(X, youngsters, axis=0) 

adults = np.where(y_t1>AGE)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X, adults, axis=0) 

lm = linear_model.LinearRegression()
error_old_tot = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    error_old_tot[:,idx] = lm.coef_
mean_pred_old_male = np.nanmean(error_old_tot, axis=1)


lm = linear_model.LinearRegression()
error_young_tot = np.empty((data_young.shape[1], NUM_REPS))
error_young_tot[:] = np.nan
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young = train_test_split(data_young, y_young)
    lm.fit(X_train, y_train)
    error_young_tot[:,idx] = lm.coef_
mean_pred_young_male = np.nanmean(error_young_tot, axis=1)


# contribution gender
print(mean_pred_old_male[-1])
print(mean_pred_young_male[-1])
#-1.3571855812663514
#-0.12319173388350388


"""
R1 - Linear with male female
"""

data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')
data_t1 = np.concatenate((data_t1, male_gender, female_gender))

dataset = 'stanford_ms_run1'
cortical_mat_ms1_t1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='t1_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2_t1, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                measure_type='t1_les')

NUM_REPS = 100
X = data_t1.T

youngsters = np.where(y_t1<AGE)
y_old = np.delete(y_t1, youngsters)
data_old = np.delete(X, youngsters, axis=0) 

adults = np.where(y_t1>AGE)
y_young = np.delete(y_t1, adults)
data_young = np.delete(X, adults, axis=0) 

lm = linear_model.LinearRegression()
error_old_tot = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    error_old_tot[:,idx] = lm.coef_
mean_pred_old_male_fem = np.nanmean(error_old_tot, axis=1)


lm = linear_model.LinearRegression()
error_young_tot = np.empty((data_young.shape[1], NUM_REPS))
error_young_tot[:] = np.nan
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young = train_test_split(data_young, y_young)
    lm.fit(X_train, y_train)
    error_young_tot[:,idx] = lm.coef_
mean_pred_young_male_fem = np.nanmean(error_young_tot, axis=1)


# contribution gender
print(mean_pred_old_male_fem[-2:])
print(mean_pred_young_male_fem[-2:])
#[-0.76343474  0.80447846]
#[-0.12719351  0.09352947]


"""
CT - Linear
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
X = data_ct.T

youngsters = np.where(y_ct<AGE)
y_old = np.delete(y_ct, youngsters)
data_old = np.delete(X, youngsters, axis=0) 

adults = np.where(y_ct>AGE)
y_young = np.delete(y_ct, adults)
data_young = np.delete(X, adults, axis=0) 

lm = linear_model.LinearRegression()
error_old_tot = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    error_old_tot[:,idx] = lm.coef_
mean_pred_old_ct = np.nanmean(error_old_tot, axis=1)
#np.count_nonzero(lm.coef_)

lm = linear_model.LinearRegression()
error_young_tot = np.empty((data_young.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young  = train_test_split(data_young, y_young)
    lm.fit(X_train, y_train)
    error_young_tot[:,idx] = lm.coef_
mean_pred_young_ct = np.nanmean(error_young_tot, axis=1)



"""
CT - Linear with male
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')
data_ct = np.concatenate((data_ct, male_gender))

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
X = data_ct.T

youngsters = np.where(y_ct<AGE)
y_old = np.delete(y_ct, youngsters)
data_old = np.delete(X, youngsters, axis=0) 

adults = np.where(y_ct>AGE)
y_young = np.delete(y_ct, adults)
data_young = np.delete(X, adults, axis=0) 

lm = linear_model.LinearRegression()
error_old_tot = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    error_old_tot[:,idx] = lm.coef_
mean_pred_old_ct_male = np.nanmean(error_old_tot, axis=1)
#np.count_nonzero(lm.coef_)

lm = linear_model.LinearRegression()
error_young_tot = np.empty((data_young.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young  = train_test_split(data_young, y_young)
    lm.fit(X_train, y_train)
    error_young_tot[:,idx] = lm.coef_
mean_pred_young_ct_male = np.nanmean(error_young_tot, axis=1)


# contribution gender
print(mean_pred_old_ct_male[-1])
print(mean_pred_young_ct_male[-1])
#0.5719272997643793
#-0.4434284639630638


"""
CT - Linear with male female
"""

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')
data_ct = np.concatenate((data_ct, male_gender, female_gender))

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
X = data_ct.T

youngsters = np.where(y_ct<AGE)
y_old = np.delete(y_ct, youngsters)
data_old = np.delete(X, youngsters, axis=0) 

adults = np.where(y_ct>AGE)
y_young = np.delete(y_ct, adults)
data_young = np.delete(X, adults, axis=0) 

lm = linear_model.LinearRegression()
error_old_tot = np.empty((data_old.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_old = train_test_split(data_old, y_old)
    lm.fit(X_train, y_train)
    error_old_tot[:,idx] = lm.coef_
mean_pred_old_ct_male_fem = np.nanmean(error_old_tot, axis=1)
#np.count_nonzero(lm.coef_)

lm = linear_model.LinearRegression()
error_young_tot = np.empty((data_young.shape[1], NUM_REPS))
for idx in range(NUM_REPS):
    X_train, X_test, y_train, y_test_young  = train_test_split(data_young, y_young)
    lm.fit(X_train, y_train)
    error_young_tot[:,idx] = lm.coef_
mean_pred_young_ct_male_fem = np.nanmean(error_young_tot, axis=1)


# contribution gender
print(mean_pred_old_ct_male_fem[-2:])
print(mean_pred_young_ct_male_fem[-2:])
#[ 0.26686642 -0.26635706]
#[-0.23227805  0.2312307 ]



mean_pred_old
mean_pred_old_male
mean_pred_old_male_fem
mean_pred_old_ct
mean_pred_old_ct_male
mean_pred_old_ct_male_fem


mean_pred_young
mean_pred_young_male
mean_pred_young_male_fem
mean_pred_young_ct
mean_pred_young_ct_male
mean_pred_young_ct_male_fem

# error_old_pca_ct error_old_pca_r1
from scipy.stats import ttest_ind

t, p = ttest_ind(np.abs(mean_pred_old), np.abs(mean_pred_old_male))
#sns.boxplot(data=[np.abs(mean_pred_old) , np.abs(mean_pred_old_male)], color='b')
t, p = ttest_ind(np.abs(mean_pred_old), np.abs(mean_pred_old_male_fem))
t, p = ttest_ind(np.abs(mean_pred_old_ct), np.abs(mean_pred_old_ct_male))
t, p = ttest_ind(np.abs(mean_pred_old_ct), np.abs(mean_pred_old_ct_male_fem))

t, p = ttest_ind(np.abs(mean_pred_young), np.abs(mean_pred_young_male))
t, p = ttest_ind(np.abs(mean_pred_young), np.abs(mean_pred_young_male_fem))
t, p = ttest_ind(np.abs(mean_pred_young_ct), np.abs(mean_pred_young_ct_male))
t, p = ttest_ind(np.abs(mean_pred_young_ct), np.abs(mean_pred_young_ct_male_fem))




# statsmodel
data_t1, y_t1, subs_t1 = load_cort(measure = 'r1', cort='midgray')
data_t1 = np.concatenate((data_t1, male_gender, female_gender))
X = data_t1.T
y = y_t1

#data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')
#data_ct = np.concatenate((data_ct, male_gender, female_gender))
#X = data_ct.T
#y = y_ct

# statsmodel
x2 = sm.add_constant(X*100)
models = sm.OLS(y,x2)
result = models.fit()
print(result.summary())