#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 09:44:12 2019

@author: asier.erramuzpe
"""

import numpy as np
from src.data.make_dataset import (load_cort,
                                   load_dataset,
                                   load_cort_areas,
                                   )
from src.visualization.visualize import (plot_areas,
                                         make_freeview_fig,
                                         )
areas = load_cort_areas()
from scipy.stats import pearsonr

#"""
#CT
#"""
#dataset = 'huji'
#cortical_parc = 'volume' ## 'volume'
#data_matrix_imputed, age_list, subs, area_names = load_dataset(dataset, cortical_parc=cortical_parc, measure_type='t1')
    
"""
T1
"""  
dataset = 'huji'
cortical_parc = 'midgray' ## 'volume'
data_t1, y_t1, subs_t1, areas = load_dataset(dataset, cortical_parc=cortical_parc, measure_type='r1')

subjects_to_delete = [1, 3, 4, 5, 17, 24]
data_t1 = np.delete(data_t1, subjects_to_delete, axis=1)
y_t1 = np.delete(y_t1, subjects_to_delete)
#subs_t1 = np.delete(subs_t1, subjects_to_delete)
#sex= [0, 1, 1, 1, 1, 1,
# 1, 0, 1, 1, 1, 1, 
# 0, 1, 1, 0, 1, 1, 
# 1, 1, 1, 1, 0, 0, 
# 0, 0, 1, 0, 1, 1, 
# 0, 1, 1, 0] # 1-male. 23 Male. Young:10 Male


dataset = 'huji'
cortical_parc = 'volume' 
data_ct, y_ct, subs_ct, areas = load_dataset(dataset, cortical_parc=cortical_parc, measure_type='t1')
"""
This script builds several figures and 2 gifs for HUJI DATASET:
    - Maturation process of T1, from max at age of 8 to min (maturation)
    - Aging proccess of T1, from min (maturation) to max (aging)
    - GIF of T1 aging 8-80
    - Aging process of CT (decay) and GIF

TODO: Expand with individual sns.regplots in case we want to build a more complete figure
"""


"""
2 figures. Changes in maturation, changes in aging.
"""
NUM_AREAS = 148

change_young = np.zeros(NUM_AREAS)
change_old = np.zeros(NUM_AREAS)
mean_min_t1_age = np.zeros(NUM_AREAS)

bins = np.linspace(20, 80, 3)
y_binned = np.digitize(y_t1, bins)

mean_young = np.zeros(data_t1.shape[0])
mean_old = np.zeros(data_t1.shape[0])
for xbin in np.unique(y_binned):
    idx_binx = np.where(y_binned == xbin)[0]
    if xbin==1:
        mean_young = np.mean(data_t1[:, idx_binx], axis=1)
    else:
        mean_old = np.mean(data_t1[:, idx_binx], axis=1)

change_young_old = mean_old - mean_young

plot_area_values(change_young_old, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_change_gm_young_HUJI_mean.jpg')


### most mature - non related to MS (frontal areas, some occipital related)
figures = []
idx_young = change_young_old.argsort()[::-1]
for idx in idx_young:
    plt.figure()
    sns.regplot(x=y_t1, y=data_t1[idx, :], order=1, color='blue')
    plt.title(areas[idx])
    plt.ylim((1.1, 1.6)) # plt.ylim((0.11, 0.24)) for TV
    ax = plt.title(areas[idx] + ' ' + str(idx))
    fig = ax.get_figure()
    figures.append(fig)
    plt.close()

multipage(opj('/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/',
              'regplot_most_mature_t1_huji.pdf'),
              figures,
              dpi=100)

areas_notdeg2 = np.zeros(NUM_AREAS)
areas_notdeg2[np.where(areas_notdeg2==0)] = np.nan
areas_notdeg2[np.where(change_young_old<0)] = 1
make_freeview_fig(areas_notdeg2, 'areas_not_deg2_huji')

"""
For CT
"""
change_young = np.zeros(NUM_AREAS)
change_old = np.zeros(NUM_AREAS)
mean_min_t1_age = np.zeros(NUM_AREAS)

bins = np.linspace(20, 80, 3)
y_binned = np.digitize(y_ct, bins)

mean_young = np.zeros(data_ct.shape[0])
mean_old = np.zeros(data_ct.shape[0])
for xbin in np.unique(y_binned):
    idx_binx = np.where(y_binned == xbin)[0]
    if xbin==1:
        mean_young = np.mean(data_ct[:, idx_binx], axis=1)
    else:
        mean_old = np.mean(data_ct[:, idx_binx], axis=1)

change_young_old_ct = mean_young- mean_old 

plot_area_values(change_old_ct, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_change_gm_ct_HUJI_mean.jpg')




for area in range(NUM_AREAS):
    p_ct = np.polyfit(y_ct, data_ct[area, :], deg=1)
    p_ct = np.poly1d(p_ct)
        
    xp_ct = np.linspace(8, 80, 73)
    max_ct_young = p_ct(8)
    max_ct_old = p_ct(80)
    
    change_old_ct[area] = p_ct(42)- max_ct_old
    
plot_area_values(change_old_ct, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/t1_change_gm_ct_HUJI_mean_42.jpg')

    
    
"""
TV
"""  
dataset = 'huji'
cortical_parc = 'midgray' ## 'volume'
data_t1, y_t1, subs_t1, areas = load_dataset(dataset, cortical_parc=cortical_parc, measure_type='tv')

subjects_to_delete = [1, 3, 4, 5, 17, 24]
data_t1 = np.delete(data_t1, subjects_to_delete, axis=1)
y_t1 = np.delete(y_t1, subjects_to_delete)
"""
This script builds several figures and 2 gifs for HUJI DATASET:
    - Maturation process of T1, from max at age of 8 to min (maturation)
    - Aging proccess of T1, from min (maturation) to max (aging)
    - GIF of T1 aging 8-80
    - Aging process of CT (decay) and GIF

TODO: Expand with individual sns.regplots in case we want to build a more complete figure
"""


"""
2 figures. Changes in maturation, changes in aging.
"""
NUM_AREAS = 148

change_young = np.zeros(NUM_AREAS)
change_old = np.zeros(NUM_AREAS)
mean_min_t1_age = np.zeros(NUM_AREAS)


bins = np.linspace(20, 80, 3)
y_binned = np.digitize(y_t1, bins)

mean_young = np.zeros(data_t1.shape[0])
mean_old = np.zeros(data_t1.shape[0])
for xbin in np.unique(y_binned):
    idx_binx = np.where(y_binned == xbin)[0]
    if xbin==1:
        mean_young = np.mean(data_t1[:, idx_binx], axis=1)
    else:
        mean_old = np.mean(data_t1[:, idx_binx], axis=1)

change_young_old = mean_old - mean_young

plot_area_values(change_young_old, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/tv_change_gm_young_HUJI_mean.jpg')







##############
##############
"""
HUJI/STANFORD DATASET COMPARISON 27 VS 67
"""
##############
##############



import numpy as np
from src.data.make_dataset import load_dataset
from src.visualization.visualize import plot_areas
areas = load_cort_areas()

NUM_AREAS = 148
areas_notdeg2_idx = [  0,   1,   4,   5,  11,  12,  13,  17,  20,  21,  22,  23,  30,  31,  33,
                       34,  36,  38,  41,  42,  46,  47,  49,  61,  62,  63,  69, 71, 74,
                       75,  79,  94,  95,  96,  97, 104, 105, 108, 115, 116, 120, 136, 137,
                       143] 
areas_notdeg2 = np.zeros(NUM_AREAS)
areas_notdeg2[areas_notdeg2_idx] = 1
"""
R1
"""  

areas = load_cort_areas()
data_t1, y_t1, subs_t1 = load_cort(measure='r1', cort='midgray')
data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')

dataset = 'huji'
cortical_parc = 'midgray' ## 'volume'
data_t1_huji, y_t1_huji, subs_t1, areas = load_dataset(dataset, cortical_parc=cortical_parc, measure_type='r1')
subjects_to_delete = [1, 3, 4, 5, 17, 24]
data_t1_huji = np.delete(data_t1_huji, subjects_to_delete, axis=1)
y_t1_huji = np.delete(y_t1_huji, subjects_to_delete)

# np.mean(y_t1_huji[np.where(y_binned == 1)]) MEAN YOUNG = 26.93
# np.mean(y_t1_huji[np.where(y_binned == 2)]) MEAN OLD = 67.66

dataset = 'huji'
cortical_parc = 'volume' 
data_ct_huji, y_ct_huji, subs_ct, areas = load_dataset(dataset, cortical_parc=cortical_parc, measure_type='t1')
data_ct_huji = np.delete(data_ct_huji, subjects_to_delete, axis=1)
y_ct_huji = np.delete(y_ct_huji, subjects_to_delete)

# change t1 huji
bins = np.linspace(20, 80, 3)
y_binned = np.digitize(y_t1_huji, bins)
mean_young = np.zeros(data_t1_huji.shape[0])
mean_old = np.zeros(data_t1_huji.shape[0])
for xbin in np.unique(y_binned):
    idx_binx = np.where(y_binned == xbin)[0]
    if xbin==1:
        mean_young = np.mean(data_t1_huji[:, idx_binx], axis=1)
    else:
        mean_old = np.mean(data_t1_huji[:, idx_binx], axis=1)

change_young_old_huji = mean_young - mean_old 

# change ct huji
bins = np.linspace(20, 80, 3)
y_binned = np.digitize(y_ct_huji, bins)
mean_young = np.zeros(data_ct_huji.shape[0])
mean_old = np.zeros(data_ct_huji.shape[0])
for xbin in np.unique(y_binned):
    idx_binx = np.where(y_binned == xbin)[0]
    if xbin==1:
        mean_young = np.mean(data_ct_huji[:, idx_binx], axis=1)
    else:
        mean_old = np.mean(data_ct_huji[:, idx_binx], axis=1)

change_young_old_huji_ct = mean_young - mean_old 

#plot_area_values(change_young_old_huji_ct, '/ems/elsc-labs/mezer-a/Mezer-Lab/projects/code/wm-covar/reports/figures/exploratory/ct_change_gm_HUJI_mean.jpg')



# change t1 and ct stanford
change_young_old_stanford = np.zeros(NUM_AREAS)
change_young_old_stanford_ct = np.zeros(NUM_AREAS)

for area in range(NUM_AREAS):
    p = np.polyfit(y_t1, data_t1[area, :], deg=2)
    p = np.poly1d(p)
    p_ct = np.polyfit(y_ct, data_ct[area, :], deg=1)
    p_ct = np.poly1d(p_ct)
    
    xp = np.linspace(8, 80, 73)
    max_t1_young = p(26.93)
    max_t1_old = p(67.66)
    
    xp_ct = np.linspace(8, 80, 73)
    max_ct_young = p_ct(26.93)
    max_ct_old = p_ct(67.66)

    change_young_old_stanford[area] =  max_t1_young - max_t1_old
    change_young_old_stanford_ct[area] =   max_ct_young - max_ct_old


## T1
change_young_old_huji[np.where(areas_notdeg2==1)] = np.nan
change_young_old_stanford[np.where(areas_notdeg2==1)] = np.nan
c_range = [np.nanmin(np.concatenate([change_young_old_huji,change_young_old_stanford])), np.nanmax(np.concatenate([change_young_old_huji,change_young_old_stanford]))]

make_freeview_fig(change_young_old_huji, 't1_change_gm_HUJI_mean_fs', c_range)
make_freeview_fig(change_young_old_stanford, 't1_change_gm_stanford', c_range)
## Correlation between weights of 2 methods R1
bad = ~np.logical_or(np.isnan(change_young_old_huji), np.isnan(change_young_old_stanford))
r, pval = pearsonr(np.compress(bad, change_young_old_huji), np.compress(bad, change_young_old_stanford) )
print(r, pval)
sns.regplot(x=np.compress(bad, change_young_old_huji), y=np.compress(bad, change_young_old_stanford), order=1, color='blue')
plt.savefig('./reports/figures/exploratory/r1_change_huji_vs_change_stanford_r0.65.svg', format='svg')


## CT
change_young_old_huji_ct[np.where(areas_notdeg2==1)] = np.nan
change_young_old_stanford_ct[np.where(areas_notdeg2==1)] = np.nan
c_range = [np.nanmin(np.concatenate([change_young_old_huji_ct,change_young_old_stanford_ct])), np.nanmax(np.concatenate([change_young_old_huji_ct,change_young_old_stanford_ct]))]

make_freeview_fig(change_young_old_huji_ct, 'ct_change_gm_HUJI_mean_fs', c_range)
make_freeview_fig(change_young_old_stanford_ct, 'ct_change_gm_stanford_mean_fs', c_range)
## Correlation between weights of 2 methods CT
bad = ~np.logical_or(np.isnan(change_young_old_huji_ct), np.isnan(change_young_old_stanford_ct))
r, pval = pearsonr(np.compress(bad, change_young_old_huji_ct), np.compress(bad, change_young_old_stanford_ct) )
print(r, pval)
sns.regplot(x=np.compress(bad, change_young_old_huji_ct), y=np.compress(bad, change_young_old_stanford_ct), order=1, color='blue')
plt.savefig('./reports/figures/exploratory/ct_change_huji_vs_change_stanford_r0.31.svg', format='svg')
