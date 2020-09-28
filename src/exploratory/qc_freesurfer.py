#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:36:58 2020

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
import nibabel as nib

from src.data.make_dataset import (load_cort,
                                   load_cort_areas,
                                   load_dataset,
                                   )
from src.visualization.visualize import (plot_area_values,
                                         plot_areas,
                                         plot_area_values_vmin,
                                         multipage,
                                         )

def get_axial(file):
    import nibabel as nib
    
    try:
        file_data = nib.load(file).get_data()
    except:
        print('File {} does not exist'.format(file))  
        
    x, y, z = file_data.shape
    file_axial = file_data[:,  z//2, :]
    
    return file_axial


#dataset = 'stanford_2'
#data_stan, age_stan, subs_stan, _ = load_dataset(dataset)
#dataset = 'kalanit'
#data_kalanit, age_kalanit, subs_kalanit, _ = load_dataset(dataset)
dataset = 'gotlib'
data_matrix_gotlib, age_gotlib, subs_gotlib, _ = load_dataset(dataset)   

          
figures = []
for sub in subs_gotlib:                                                             
    sub_fs = sub + '_'+  dataset;
    aparc_file = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/freesurfer_subjects/' + sub_fs + '/mri/aparc.a2009s+aseg.mgz' 
    orig_file = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/freesurfer_subjects/' + sub_fs + '/mri/orig.mgz'   
    
    if os.path.exists(aparc_file) and os.path.exists(orig_file):
        axial_slice = get_axial(aparc_file)   
        axial_slice2 = get_axial(orig_file)         
        # make figure
        plt.imshow(axial_slice2, cmap='gray')
        plt.imshow(axial_slice, cmap='gray', alpha=0.2)
        ax.set_rasterized(True)
        ax = plt.title('Subject {}'.format(sub))
        fig = ax.get_figure()
        fig.savefig(opj(opj('./reports','qc', 'fs'),
                        sub+'.png'))
        plt.close()

    

