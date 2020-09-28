#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 14:47:02 2018

@author: asier.erramuzpe
"""


""" 
New extraction strategy
"""

# Extract fs data with the atlas for each subject
# Remove 0s
# for each region, fit a distribution and remove non normal data with a parameter
# test against midgray and fs


ms_subs = ['MS_501_run1', 'MS_502_run1', 'MS_504_run1', 'MS_505_run1',
       'MS_506_run1', 'MS_507_run1', 'MS_508_run1', 'MS_509_run1',
       'MS_510_run1']

higher = np.zeros_like(data_matrix_ms)
lower = np.zeros_like(data_matrix_ms)
for area_idx in range(data_old.shape[0]):
    
    data_area = data_old[area_idx, :]
    mean_area = np.mean(data_area)
    std_area = np.std(data_area)
    
    for elem in range(data_matrix_ms.shape[1]):
        if data_matrix_ms[area_idx, elem] > (mean_area + std_area):
            higher[area_idx, elem] = 1
        elif data_matrix_ms[area_idx, elem] < (mean_area - std_area):
            lower[area_idx, elem] = 1

for idx_sub, subject in enumerate(ms_subs):            
    print(subject, 'has', np.sum(lower[:,idx_sub]), 'values below mean-std',
          np.sum(higher[:,idx_sub]), 'values over mean+std')
          
    

sum_lower = np.sum(lower, axis=1)
lower_areas = area_names[np.where(sum_lower >= 5)]
out_file = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/lower'
plot_areas(lower_areas, out_file, save_nifti=True)

sum_higher = np.sum(higher, axis=1)
higher_areas = area_names[np.where(sum_higher >= 5)]
out_file = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/higher'
plot_areas(higher_areas, out_file, save_nifti=True)


#for idx_sub, sub in enumerate(subs):            
#    areas = area_names[np.where(lower[:, idx_sub] == 1)]
#    out_file = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/' + sub
#    plot_areas(areas, out_file, save_nifti=True)


higher = np.zeros_like(data_old)
lower = np.zeros_like(data_old)
for area_idx in range(data_old.shape[0]):
    
    data_area = data_old[area_idx, :]
    mean_area = np.mean(data_area)
    std_area = np.std(data_area)
    
    for elem in range(data_old.shape[1]):
        if data_old[area_idx, elem] > (mean_area + std_area):
            higher[area_idx, elem] = 1
        elif data_old[area_idx, elem] < (mean_area - std_area):
            lower[area_idx, elem] = 1
            
control_subs = subs[youngsters]
for idx_sub, sub in enumerate(control_subs):            
    print(idx_sub, sub, 'has', np.sum(lower[:,idx_sub]), 'values below mean-std',
          np.sum(higher[:,idx_sub]), 'values over mean+std')
    
#sum_lower = np.sum(lower, axis=1)
#lower_areas = area_names[np.where(sum_lower >= 8)]
#out_file = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/lower_controls'
#plot_areas(lower_areas, out_file, save_nifti=False)
#
#sum_higher = np.sum(higher, axis=1)
#higher_areas = area_names[np.where(sum_higher >= 8)]
#out_file = '/ems/elsc-labs/mezer-a/asier.erramuzpe/Desktop/higher_controls'
#plot_areas(higher_areas, out_file, save_nifti=False)
