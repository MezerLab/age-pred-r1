#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:43:27 2019

@author: asier.erramuzpe
"""
### RINGING DETECTION MODULE
# file1 = OK, file2 = RINGING, file2 = SLIGHT RINGING, file3 = SLIGHT FRONTAL MOVE

file1 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s008/mrQ_ver2/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'
file2 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s007/mrQ_ver2/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'
file3 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s046/mrQ_ver2/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'
file4 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s064/mrQ_ver2/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'
file5 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s078/mrQ_ver2/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'
file6 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s106/mrQ_ver2/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'

file1 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s008/mrQ_ver2/OutPutFiles_1/BrainMaps/TV_map.nii.gz'
file2 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s007/mrQ_ver2/OutPutFiles_1/BrainMaps/TV_map.nii.gz'
file3 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s046/mrQ_ver2/OutPutFiles_1/BrainMaps/TV_map.nii.gz'
file4 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s064/mrQ_ver2/OutPutFiles_1/BrainMaps/TV_map.nii.gz'
file5 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s078/mrQ_ver2/OutPutFiles_1/BrainMaps/TV_map.nii.gz'
file6 = '/ems/elsc-labs/mezer-a/Mezer-Lab/analysis/reading_stanford/s106/mrQ_ver2/OutPutFiles_1/BrainMaps/TV_map.nii.gz'


def get_axial(file):
    import nibabel as nib
    
    try:
        file_data = nib.load(file).get_data()
    except:
        print('File {} does not exist'.format(file))  
        
    x, y, z = file_data.shape
    file_axial = file_data[:, :, z//2]
    
    return file_axial

def plot_spectrum(img):
    from matplotlib.colors import LogNorm
    from scipy import fftpack
    
    im_fft = fftpack.fft2(img)
    # A logarithmic colormap
    plt.figure()
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title('Fourier transform')


plot_spectrum(get_axial(file1))
plot_spectrum(get_axial(file2))
plot_spectrum(get_axial(file3))
plot_spectrum(get_axial(file4))
plot_spectrum(get_axial(file5))


plt.imshow(get_axial(file1))
plt.imshow(get_axial(file2))
plt.imshow(get_axial(file3))
plt.imshow(get_axial(file4))
plt.imshow(get_axial(file5))
plt.imshow(get_axial(file6))

plot_fft2_power(file1)
plot_fft2_power(file2)
plot_fft2_power(file3)
plot_fft2_power(file4)
plot_fft2_power(file5)
plot_fft2_power(file6)


power_sum(file1)
power_sum(file2)
power_sum(file3)
power_sum(file4)
power_sum(file5)
power_sum(file6)


def power_sum(file):
    
    from scipy import fftpack
    import numpy as np
    import pylab as py
     
    image = get_axial(file)
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(image)
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift( F1 )
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2 )**2
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(psd2D)
    return np.sum(psd1D)



def plot_fft2_power(file):
    
    from scipy import fftpack
    import numpy as np
    import pylab as py
     
    image = get_axial(file)
    # Take the fourier transform of the image.
    F1 = fftpack.fft2(image)
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift( F1 )
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2 )**2
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(psd2D)
    # Now plot up both
#    py.figure(1)
#    py.clf()
#    py.imshow( np.log10( image ), cmap=py.cm.Greys)
    
    py.figure(2)
    py.clf()
    py.imshow( np.log10( psd2D ))
    
#    py.figure(3)
#    py.clf()
#    py.semilogy( psd1D )
#    py.xlabel('Spatial Frequency')
#    py.ylabel('Power Spectrum')
#    py.title(str(np.sum(psd1D)))
    py.show()


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


"""
QC reports
"""
import os
from os.path import join as opj

import numpy as np
import scipy.io as sio
from src.visualization.visualize import multipage
from dotenv import find_dotenv, load_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
analysis_data_path = os.environ.get("ANALYSIS_DATA_PATH")


def create_qc_report_mrq(dataset, file_path):
    """
    Creates a visual report with axial middle slices.
    
    dataset = dataset to choose from
    file_paths = dictionary {file: path_to_file inside mrQver2 folder}
                 as many files as wanted
    """
    
    input_path = opj(analysis_data_path,
                     dataset)
    figures = []
    for sub_idx, sub in enumerate(sorted(os.listdir(input_path))):
        print(sub)
        
        for file, file_path in file_paths.items():
            target_file = opj(input_path, sub, file_path)
            if os.path.exists(target_file):
                axial_slice = get_axial(target_file)            
                # make figure
                plt.imshow(axial_slice, cmap='gray')
                plt.clim(0, 4)
                plt.colorbar()
                ax = plt.title('Subject {}. File {}'.format(sub, file))
                fig = ax.get_figure()
                figures.append(fig)
                plt.close()
    
    output_folder = opj('./reports',
                        'qc')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    multipage(opj(output_folder,
                  'report_' + dataset + '.pdf'),
                  figures,
                  dpi=250)



# possible datasets = kalanit_stanford ms_stanford_run1 stanford_2 reading_stanford anorexia_stanford gotlib_stanford amblyopia_stanford
datasets = ['kalanit_stanford', 'ms_stanford_run1', 'stanford_2',
            'reading_stanford', 'anorexia_stanford',
            'gotlib_stanford', 'amblyopia_stanford']

file_paths = {'T1': 'mrQ_ver2/OutPutFiles_1/BrainMaps/T1_map_Wlin.nii.gz'}   

for dataset in datasets: 
    create_qc_report_mrq(dataset, file_paths)
