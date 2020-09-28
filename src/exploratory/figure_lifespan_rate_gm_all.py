#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:33:34 2019

@author: asier.erramuzpe
"""
import os
from os.path import join as opj

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from io import BytesIO
from PIL import Image

from src.data.make_dataset import (load_cort,
                                   load_cort_areas,
                                   load_dataset,
                                   )
from src.visualization.visualize import (plot_area_values,
                                         plot_areas,
                                         plot_area_values_vmin,
                                         multipage,
                                         make_freeview_fig,
                                         )
"""
This script builds several figures and gifs:
    - Maturation process of T1, TV, CT and Slope from max at age of 8 to min (maturation)
    - Aging proccess of T1, TV, CT and Slope, from min (maturation) to max (aging)
    - GIF of T1, TV, CT and Slope aging 8-80
    
    Not all T1, TV, CT and Slope have the same distributions, that's why we plot them. 

TODO: Expand with individual sns.regplots in case we want to build a more complete figure
"""

NUM_AREAS = 148
areas_notdeg2_idx = [  0,   1,   4,   5,  11,  12,  13,  17,  20,  21,  22,  23,  30,  31,  33,
                       34,  36,  38,  41,  42,  46,  47,  49,  61,  62,  63,  69, 71, 74,
                       75,  79,  94,  95,  96,  97, 104, 105, 108, 115, 116, 120, 136, 137,
                       143] 
areas_notdeg2 = np.zeros(NUM_AREAS)
areas_notdeg2[areas_notdeg2_idx] = 1


##############
##############
"""
R1 and CT
"""
##############
##############


"""
Load data and plot
"""
NUM_AREAS = 148
areas = load_cort_areas()
data_t1, y_t1, subs_t1 = load_cort(measure='r1', cort='midgray')
dataset = 'stanford_ms_run1'
cortical_mat_ms1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                  measure_type='r1_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                  measure_type='r1_les')

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')

"""
2 figures. Changes in maturation, changes in aging.
"""
rate_young = np.zeros(NUM_AREAS)
rate_old = np.zeros(NUM_AREAS)
rate_young_ct = np.zeros(NUM_AREAS)
rate_old_ct = np.zeros(NUM_AREAS)
mean_min_t1_age = np.zeros(NUM_AREAS)

for area in range(NUM_AREAS):
    p = np.polyfit(y_t1, data_t1[area, :], deg=2)
    p = np.poly1d(p)
    p_ct = np.polyfit(y_ct, data_ct[area, :], deg=1)
    p_ct = np.poly1d(p_ct)
    
    xp = np.linspace(6, 81, 76)
    max_t1_young = p(6)
    max_t1_old = p(81)
    min_t1 = np.max(p(xp))
    
    xp_ct = np.linspace(6, 81, 76)
    max_ct_young = p_ct(6)
    max_ct_old = p_ct(81)

    idx_min_t1 = xp[np.argmax(p(xp))]

    rate_young[area] = (min_t1 - max_t1_young) / (idx_min_t1-6)
    rate_old[area] = (min_t1 - max_t1_old) / (81 - idx_min_t1)
    mean_min_t1_age[area] = idx_min_t1
    
    rate_young_ct[area] = np.abs((p_ct(idx_min_t1) - max_ct_young)) / (idx_min_t1-6) 
    rate_old_ct[area] = np.abs((max_ct_old - p_ct(idx_min_t1))) / (81 - idx_min_t1)

rate_young[np.where(areas_notdeg2==1)] = np.nan
rate_old[np.where(areas_notdeg2==1)] = np.nan
rate_young_ct[np.where(areas_notdeg2==1)] = np.nan
rate_old_ct[np.where(areas_notdeg2==1)] = np.nan
mean_min_t1_age[np.where(areas_notdeg2==1)] = np.nan


#plot_area_values(rate_young, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_rate_gm_young.jpg')
#plot_area_values(rate_old, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_rate_gm_old.jpg')
c_range = [0, np.nanmax(np.concatenate([rate_young,rate_old]))]
make_freeview_fig(rate_young, 'R1_rate_gm_young_fs', c_range)
make_freeview_fig(rate_old, 'R1_rate_gm_old_fs', c_range)

#plot_area_values(rate_young_ct, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_rate_gm_young_ct.jpg')
c_range = [0, np.nanmax(np.concatenate([rate_young_ct,rate_old_ct]))]
#plot_area_values(rate_old_ct, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_rate_gm_old_ct.jpg')
make_freeview_fig(rate_young_ct, 'ct_rate_gm_young_fs', c_range)
make_freeview_fig(rate_old_ct, 'ct_rate_gm_old_fs', c_range)

#mean_min_t1_age[np.where(mean_min_t1_age > 80)] = np.nan
#mean_min_t1_age[np.where(mean_min_t1_age < 32)] = np.nan
#plot_area_values_vmin(mean_min_t1_age, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/maturation_regions_t1.jpg', vmin=20, vmax=60, title='maturation_t1')
make_freeview_fig(mean_min_t1_age, 'maturation_regions_t1_fs')


#plot_areas(areas[np.where(change_young == 0)], '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_change_gm_young_nochange.jpg', save_nifti=False)
## TODO: Review what happens with areas that have minimums in late ages (decay process)
#plot_areas(areas[np.where(change_old == 0)], '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_change_gm_old_nochange.jpg', save_nifti=False)

#med_age = np.median(mean_min_t1_age)  # 42
# plt.plot(xp, p(xp))
# plt.scatter(y_t1, data_t1[area, :])



##############
##############
"""
T1  
"""
##############
##############

"""
Load data and plot
"""
NUM_AREAS = 148
areas = load_cort_areas()
data_t1, y_t1, subs_t1 = load_cort(measure='t1', cort='midgray')
dataset = 'stanford_ms_run1'
cortical_mat_ms1, age_ms1, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                  measure_type='t1_les')
dataset = 'stanford_ms_run2'
cortical_mat_ms2, age_ms2, subs, _ = load_dataset(dataset, cortical_parc='midgray',
                                                  measure_type='t1_les')

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')

"""
2 figures. Changes in maturation, changes in aging.
"""
YEARS = 25

rate_young = np.zeros(NUM_AREAS)
rate_old = np.zeros(NUM_AREAS)
rate_young_ct = np.zeros(NUM_AREAS)
rate_old_ct = np.zeros(NUM_AREAS)
mean_min_t1_age = np.zeros(NUM_AREAS)

for area in range(NUM_AREAS):
    p = np.polyfit(y_t1, data_t1[area, :], deg=2)
    p = np.poly1d(p)
    p_ct = np.polyfit(y_ct, data_ct[area, :], deg=1)
    p_ct = np.poly1d(p_ct)
    
    xp = np.linspace(8, 80, 73)
    idx_min_t1 = xp[np.argmin(p(xp))]
    
    max_t1_young = p(idx_min_t1 - YEARS)
    max_t1_old = p(idx_min_t1 + YEARS)
    min_t1 = np.min(p(xp))
    
    xp_ct = np.linspace(8, 80, 73)
    max_ct_young = p_ct(idx_min_t1 - YEARS)
    max_ct_old = p_ct(idx_min_t1 + YEARS)

    idx_min_t1 = xp[np.argmin(p(xp))]

    rate_young[area] = (max_t1_young - min_t1) / YEARS
    rate_old[area] = (max_t1_old - min_t1) / YEARS
    mean_min_t1_age[area] = idx_min_t1
    
    rate_young_ct[area] = (max_ct_young - p_ct(idx_min_t1)) / YEARS
    rate_old_ct[area] = (p_ct(idx_min_t1)- max_ct_old) / YEARS

# remove areas not following U shape
rate_young[np.where(areas_notdeg2==1)] = np.nan
rate_old[np.where(areas_notdeg2==1)] = np.nan
rate_young_ct[np.where(areas_notdeg2==1)] = np.nan
rate_old_ct[np.where(areas_notdeg2==1)] = np.nan

#
#plot_area_values(rate_young, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_rate15_gm_young.jpg')
#plot_area_values(rate_old, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_rate15_gm_old.jpg')
#make_freeview_fig(np.abs(rate_young), 't1_rate_' + str(YEARS) + '_gm_young_fs')
make_freeview_fig(np.abs(rate_old), 't1_rate_' + str(YEARS) + '_old_fs')

#plot_area_values(rate_young_ct, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_rate15_gm_young_ct.jpg')
#plot_area_values(rate_old_ct, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_rate15_gm_old_ct.jpg')
#make_freeview_fig(rate_young_ct, 'ct_rate' + str(YEARS) + '_gm_young_fs')
make_freeview_fig(rate_old_ct, 'ct_rate_' + str(YEARS) + '_old_fs')
#
#mean_min_t1_age[np.where(mean_min_t1_age > 60)] = np.nan
#mean_min_t1_age[np.where(mean_min_t1_age < 20)] = np.nan
#plot_area_values_vmin(mean_min_t1_age, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/maturation_regions_t1.jpg', vmin=20, vmax=60, title='maturation_t1')
#make_freeview_fig(mean_min_t1_age, 'maturation_regions_t1_fs')



