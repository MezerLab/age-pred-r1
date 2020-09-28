#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:43:49 2019

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

def plot_descriptor_group(desc_num, data1, y1, data2, y2, area_names):
    # look subs_old and plot accordingly
    plt.figure()
    p = np.polyfit(y1, data1[desc_num, :], deg=1)
    p = np.poly1d(p)
    xp = np.linspace(8, 42, 35)
    plt.plot(xp, p(xp))
    plt.scatter(y1, data1[desc_num, :])
    plt.scatter(y2, data2[desc_num,:])

    plt.xlabel('Age of subject')
    plt.ylabel('qMRI value')
    plt.legend(['1st deg fit', 'Con', 'MS'])
    plt.title((area_names[desc_num], str(desc_num)))
    return  

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

#data_t1 = np.delete(data_t1, areas_to_remove_idx, axis=0) 
#cortical_mat_ms1 = np.delete(cortical_mat_ms1, areas_to_remove_idx, axis=0) 
#cortical_mat_ms2 = np.delete(cortical_mat_ms2, areas_to_remove_idx, axis=0) 

idx = np.where((y_t1 > 40) & (y_t1 < 62))

mean_vals_t1 = np.mean(data_t1[:, idx[0]], axis=1)
mean_vals_ms = np.mean(np.hstack((cortical_mat_ms1, cortical_mat_ms2)), axis=1)

diff_ct = mean_vals_t1 - mean_vals_ms # contrarily to T1, here it's CT healthy supposed to be higher
#make_freeview_fig(diff, 'ms_ct_diff_fs')
diff_ct = mean_vals_t1 - mean_vals_ms
diff_ct[np.where(areas_notdeg2==1)] = np.nan
make_freeview_fig(diff_ct, 'ms_ct_diff')

"""
most loss areas
"""
idx_loss = diff.argsort()[-50:] # ordered from bottom to top (is better to have higher CT)
areas[idx_loss]
plot_areas(areas[idx_loss], '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/ms_most_loss_10_areas_ct.jpg', save_nifti=False)

val_mat = np.zeros(areas.shape)
val_mat[idx_loss] = 1
val_mat[np.where(val_mat==0)] = np.nan
make_freeview_fig(val_mat, 'ms_ct_most_loss_50_areas_fs')
#"""
#most gain areas
#"""
#idx_gain = diff.argsort()[:10][::-1]
#areas[idx_gain]
#plot_areas(areas[idx_gain], '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/ms_most_gain_10_areas_ct.jpg', save_nifti=False)
#
#plt.hist(diff)
#
#for idx in idx_loss:
#    plot_descriptor_group(idx, data_t1, y_t1, np.hstack((cortical_mat_ms1, cortical_mat_ms2)) , np.hstack((age_ms1, age_ms2)), areas)


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

diff = mean_vals_t1 - mean_vals_ms
diff[np.where(areas_notdeg2==1)] = np.nan
diff[ np.where(diff<0)] = np.nan
make_freeview_fig(diff, 'ms_R1_diff_fs')

"""
most loss areas
"""
idx_loss = diff.argsort()[-50:][::-1] # ordered from bottom to top (is better to have lower T1)
areas[idx_loss]
plot_areas(areas[idx_loss], '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/ms_R1_most_loss_50_areas.jpg', save_nifti=False)

val_mat = np.zeros(areas.shape)
val_mat[idx_loss] = 1
val_mat[np.where(val_mat==0)] = np.nan
make_freeview_fig(val_mat, 'ms_R1_most_loss_50_areas_fs')

#"""
#most gain areas
#"""
#idx_gain = diff.argsort()[:10][::-1]
#areas[idx_gain]
#plot_areas(areas[idx_gain], '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/ms_most_gain_10_areas.jpg', save_nifti=False)
#
#plt.hist(diff)


"""
TV
"""
data_tv, y_tv, subs_tv = load_cort(measure='tv', cort='midgray')

dataset = 'stanford_ms_run1'
cortical_mat_ms1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                  measure_type='tv_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                  measure_type='tv_les')

idx = np.where((y_tv > 40) & (y_tv < 62))

mean_vals_tv = np.mean(data_tv[:, idx[0]], axis=1)
mean_vals_ms = np.mean(np.hstack((cortical_mat_ms1, cortical_mat_ms2)), axis=1)

diff = mean_vals_ms - mean_vals_tv

"""
most loss areas
"""
idx_loss = diff.argsort()[-10:][::-1]
areas[idx_loss]
plot_areas(areas[idx_loss], '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/ms_most_loss_10_areas_tv.jpg', save_nifti=False)

"""
most gain areas
"""
idx_gain = diff.argsort()[:10][::-1]
areas[idx_gain]
plot_areas(areas[idx_gain], '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/ms_most_gain_10_areas_tv.jpg', save_nifti=False)

plt.hist(diff)