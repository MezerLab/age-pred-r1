#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:53:30 2019

@author: asier.erramuzpe
"""


NUM_AREAS = 148
areas = load_cort_areas()
data_t1, y_t1, subs_t1 = load_cort(measure='r1', cort='midgray')

data_ct, y_ct, subs_ct = load_cort(measure = 't1', cort='volume')


ctx_rh_G_occipital_middle = 93 # 38 default color (blue) 
ctx_rh_G_parietal_sup = 100 # 41 cornflowerblue
ctx_rh_G_temp_sup-Lateral = 107 # 57 yellowgreen or olive
ctx_rh_G_front_middle = 88 # 54 mediumseagreen

region = 88
mean_min_t1_age[region]
#val_mat = np.zeros(148)
#val_mat[np.where(val_mat==0)] = np.nan
#val_mat[region] = 1
#make_freeview_fig(val_mat, 'region_test'+str(region)+'.png')


# R1 fit
plt.figure()
sns.regplot(x=y_t1, y=data_t1[region, :], order=2, color='mediumseagreen')
plt.title(areas[region])
plt.ylim((0.6, 0.85)) # plt.ylim((0.11, 0.24)) for TV
plt.savefig('./reports/poster_ohbm/r1_'+areas[region]+'.svg', format='svg')

# ct fit
plt.figure()
sns.regplot(x=y_ct, y=data_ct[region, :], order=1, color='mediumseagreen')
plt.title(areas[region])
plt.ylim((1.8, 4)) 
plt.savefig('./reports/poster_ohbm/ct_'+areas[region]+'.svg', format='svg')





array([  2,  22,  54,  59,  71,  76,  93, 110, 123])
val_mat = np.zeros(148)
val_mat[np.where(val_mat==0)] = np.nan
val_mat[np.array([  2,  22,  54,  59,  71,  76,  93, 110, 123])] = 1
make_freeview_fig(val_mat, 'test.png')



val_mat = np.zeros(148)
val_mat[np.where(val_mat==0)] = np.nan
val_mat[np.where(areas_notdeg2==1)] = 1
make_freeview_fig(val_mat, 'test_notdeg2.png')
areas_notdeg2