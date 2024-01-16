# -*- coding: utf-8 -*-
"""
LiVaS pipeline functions.

Usage:
    import os, sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    from LiVaS_helper import *

License:
    GNU GPL 2.0
 
Revision: Dec 2, 2023
"""

import numpy as np
from tkinter import Tk, filedialog#, messagebox
import SimpleITK as sitk
import scipy.ndimage as ndimage
import random

import glob
import pydicom
import os
from typing import List, Tuple
import faiss
import string

def folderChooser() -> str:
    """
    DICOM folder chooser.
    """      
    root = Tk() # pointing root to Tk() to use it as Tk() in program.
    root.withdraw() # Hides small tkinter window.
    root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.  
    #dicom_dir = filedialog.askdirectory(initialdir="~/Documents/") # Returns opened path as str
    dicom_dir = filedialog.askdirectory(parent=root, # will remember the last opened dir
                            title='CHOSE A FOLDER WITH THE PHASE SERIES DATA:')
    root.update()
    root.destroy()
    if dicom_dir:
        return(dicom_dir)
    else:
        return("break")
    
def anyMatch(x: str, match_list: List[str]) -> bool:
    """
    Return True if any element of the match_list is in x.
    """
    for key in match_list:
        if key in x:
            return True
    return False
    
def sortDcmSeriesByPhase(strings_to_sort: List[str],
                         sort_order_list: List[str]) -> List[str]:
    """
    Sort strings_to_sort elements based on the order of keys from 
    sort_order_list.
    """
    # sort_order = ["PRE","ARTERIAL", "PVP.M", "PVP", "DELAYED.M","DELAYED.",
    #               "HBP.M","HBP.","SEG.Portal","SEG.Hepatic","SEG.Arteries","aDummy"]
    # aDummy element can be replaced with additional phase search terms
    
    #---------------- sorting function ----------------            
    def sortFun(x):
        for j, phrase in enumerate(sort_order_list):
            if phrase in x:
                #print(j, phrase)
                return (j, x)
        return (len(sort_order_list), x) # Handles strings not matching any phrase in sort_order_list
    #--------------------------------------------------        
    return(sorted(strings_to_sort, key=sortFun))

def load_dicoms(dicom_dir: str):
    """
    A function that loads dicom files into a 4D phase array.

    Parameters
    ----------
    dicom_dir : str
        A path to directory with all phase subdirectories (with dicom files).
    """
    print("Loading DICOMs from " + dicom_dir)
    
    # dicom filenames
    dicom_files = glob.glob(os.path.join(dicom_dir, '*.dcm'))
    
    # read in all dicom files
    dicoms = [pydicom.read_file(dicom_file) for dicom_file in dicom_files]
    
    # Keep only those with pixel data
    dicoms = [dicom for dicom in dicoms if 'PixelData' in dicom]
    
    # sort dicoms by slices
    slice_sorts = np.argsort([dicom.SliceLocation for dicom in dicoms])
    dicoms = [dicoms[slice_sort] for slice_sort in slice_sorts]
    
    # read in pixel array
    image = np.transpose([dicom.pixel_array for dicom in dicoms],
                         axes = (1,2,0) )
    
    # MZ: force background to be zero (shift the min to zero)
    #  Bias field correction will fail if background is not 0
    image = image - image.min()
    
    # Prevent from loading empty series (e.g. all-zeros images)
    assert np.unique(image).size > 1,\
        """Image series has no more than one intensity value!!! 
        This could be caused by failed registration of phase images."""
    
    # store slope/orientation to standardize image
    slope       = np.float32(dicoms[1].ImagePositionPatient[2]) - \
        np.float32(dicoms[0].ImagePositionPatient[2])
    orientation = np.float32(dicoms[0].ImageOrientationPatient[4])
    
    # standardize image position
    if slope < 0: image = np.flip(image, -1)  # enforce feet first axially
    if orientation < 0: image = np.flip(image, 0)  # enforce supine orientation
    
    return ( dicoms, image, slope, orientation )

def correctBias(phase_arr_4d: np.ndarray, mask_arr_3d: np.ndarray = None):
    """
    Correct each of the phase 3D images for low frequency inhomogeneities.

    Parameters
    ----------
    phase_arr_4d : np.ndarray
        Input 4D phase array.
    mask_arr_3d : np.ndarray, optional
        If a 3D mask is not provided, bias field will be computed for the whole
        image. This will result in longer computation, and here in suboptimal
        bias field generation (since the liver masks were applied onto images).
    """
    if np.any(mask_arr_3d) == None: # grab one of the phase 3d images and set it to 1
        print("Bias correction will be applied to the whole volume.")
        mask_arr_3d = np.ones_like(phase_arr_4d[:,:,:,1])#.astype('float64')
        #mask_arr_3d[:] = 1
    phase_arr_4d_corrected = np.zeros_like(phase_arr_4d).astype(np.float32)
    bias_field_arr_4d = np.zeros_like(phase_arr_4d).astype(np.float32)
    for j in range(phase_arr_4d.shape[3]): # for each phase
        print("\tCorrecting phase ", j)
        img = sitk.GetImageFromArray(phase_arr_4d[:,:,:,j])
        img = sitk.Cast(img, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetNumberOfControlPoints((4,4,4)) # 4 in x,y, and z direction
        corrector.SetBiasFieldFullWidthAtHalfMaximum(.40) # default is .15
        maskImage = sitk.GetImageFromArray(mask_arr_3d.astype('uint8'))
        corrected_image = corrector.Execute(img, maskImage)
        phase_arr_4d_corrected[:,:,:,j] = sitk.GetArrayFromImage(corrected_image)
        
        bias_field = img / corrected_image # this is quasi bias field, also could produce NaNs
        bias_field_arr_4d[:,:,:,j] = sitk.GetArrayFromImage(bias_field)
        # phase_corrected_arr.round().astype('uint16')
    return(phase_arr_4d_corrected, bias_field_arr_4d)

def normalize_data(data: np.ndarray,
                   targetMin: float = 0, targetMax: float = 1) -> np.ndarray:
    """
    Rescales the array between targetMin and targetMax.

    Parameters
    ----------
    data : np.ndarray
    targetMin : float, optional; the default is 0.
    targetMax : float, optional; the default is 1.

    Returns
    -------
    z: np.ndarray
    """
    z = (data - np.min(data)) / (np.max(data) - np.min(data))
    z = z * (targetMax - targetMin) + targetMin
    return(z)

def inverse_data(data: np.ndarray) -> np.ndarray:
    """
    Example: [0,3,2,1] would become [3,0,1,2]

    Parameters
    ----------
    data : np.ndarray
        Input data array.

    Returns
    -------
    data : np.ndarray
        Inverted array.

    """
    return(-data + np.max(data))

def get_spatial_weight(num_of_clusters: int) -> float:
     w = num_of_clusters ** (-4/num_of_clusters)
     print("Spatial feature weight = %0.2f" % w)
     return w

def cluster_phases(phase_array_: np.ndarray, 
                   liv_ind: Tuple[np.ndarray, np.ndarray, np.ndarray], 
                   number_clusters: int = 10,
                   number_phases: int = 4) -> np.ndarray:
    """
    K-means clustering.

    Parameters
    ----------
    phase_array_ : np.ndarray
        A 4D phase array.
    liv_ind : Tuple[np.ndarray, np.ndarray, np.ndarray]
        A 3D indices of a liver mask.
    number_clusters : int, optional
        K, number of clusters to be produced. The default is 10.
    number_phases : int, optional
        Number of phases to be used in cluster computation. The default is 4 
        (pre, arterial, pvp, delayed). 

    Returns
    -------
    clust : np.ndarray
        A 3D clusters array.

    """
        
    # extract only pixel values in the liver
    pixel_values = phase_array_[liv_ind]
    print("Pixel values shape to be clustered = ", pixel_values.shape)
    
    # use only specified number of phases (4 is default)
    pixel_values = pixel_values[:, 0:number_phases]
    
    # add 3 columns: spatial information (indices in x, y and z direction)
    print("Adding spatial info...")
    pixel_values = np.append(pixel_values, np.array(liv_ind).T, 1)
    
    # rescale all phases to values in [0,1] range
    print("Rescaling phases...")
    pixel_values = np.apply_along_axis(normalize_data, axis = 0, arr = pixel_values)
    
    # reduce magnitude of spatial info (last 3 columns of the array)
    pixel_values[:, -3:] = pixel_values[:, -3:] * get_spatial_weight(number_clusters)
    # add an inverse of ARTERIAL phase as another column to a phases' matrix
    pixel_values = np.append(pixel_values, inverse_data(pixel_values[:,1]).reshape((-1, 1)), 1)
    
    # Perform clustering
    print("Clustering using %s centroids..." % number_clusters)
    #random.seed(123)   
    kmeans = faiss.Kmeans(d=pixel_values.shape[1], k=number_clusters,
                          niter=300, nredo=25, verbose=False)
    pixel_values = pixel_values.copy(order='C')
    kmeans.train(pixel_values.astype(np.float32)) # fit
    labels = kmeans.index.search(pixel_values.astype(np.float32), 1)[1] # predict
    clust = np.zeros(phase_array_.shape[0:3], dtype=np.uint8)
    clust[liv_ind] = labels.flatten() + 1
    
    # return cluster labels
    return clust

def resize_array_interpolation(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Resize an array to the target shape using interpolation, only if sizes differ.

    Parameters
    ----------
    arr (np.ndarray): The input array to be resized.
    target_shape (Tuple[int, int, int]): The target dimensions to resize to.

    Returns
    -------
    np.ndarray: The resized array.
    """
    # Check if the current array shape matches the target shape
    if arr.shape != target_shape:
        print(f"Resizing from {arr.shape} to {target_shape}.")
        # Calculate the scale factors for each dimension
        scale_factors = [t/s for s, t in zip(arr.shape, target_shape)]
        # Use zoom for resampling
        return zoom(arr, scale_factors)
    else:
        print(f"No resizing needed for array with shape {arr.shape}.")
        return arr

def voxel_clustering_pipeline(dicom_dir: str, sort_order_key_list: List[str]):
    """
    Load the images, perform bias field correction, then for each K size apply 
    K-Means clustering on 4D phase images and save resulting arrays to disk as 
    compressed numpy arrays.

    Parameters
    ----------
    dicom_dir : str
        A path to directory with all phase subdirectories (with dicom files).
    sort_order_key_list : List[str]
        List containing search kewords to be used in sorting phase series.
        The order of search keys in the list should reflect the AP timing, and
        should contain substrings of the relevant phase-directories' names.
        An example list: ["PRE","ARTERIAL", "PVP", "PVP", "DELAYED"]. The 4 
        string here should be contained in the phase series direcotries' names.

    Returns
    -------
    None.

    """
    print("DICOM directory = %s \n" % dicom_dir)
    dicom_series_list = os.listdir(dicom_dir)
    
    # list comprehension to keep only desired series (subdirectories)
    dicom_series_list = [ x for x in dicom_series_list if anyMatch(x, sort_order_key_list)]
    
    # get sorted series names
    dicom_series_list = sortDcmSeriesByPhase(dicom_series_list,
                                             sort_order_key_list)
    print("Sorted series list: \n", *dicom_series_list, "\n", sep = "\n")
    
    # make sure there is only one Patient Name in the DICOM directory
    assert len(
        list(np.unique([x.split(sep='_')[0] for x in dicom_series_list]))) == 1,\
        'More than one Patient Name detected in the directory!'
    
    # load phase images into a 4D array
    images = [load_dicoms(os.path.join(dicom_dir, phase_dir))[1]
              for phase_dir in dicom_series_list]
    
    # Target shape is the shape of the first image array
	target_shape = images[0].shape

	# Resize image arrays if necessary
	images = [resize_array_interpolation(arr, target_shape) for arr in images]

	# Combine into a 4D array
	images = np.stack(images)
    print("Loaded 4D image array shape = %s \n" % str(images.shape))
    
    # transpose the image array
    phase_array = np.transpose(images, (1,2,3,0))
    print("Transposed 4D image array shape = %s \n" % str(phase_array.shape))
    
    # set the phase array to read only
    phase_array.setflags(write=False)
    np.savez_compressed(dicom_dir + '/phase_array', phase_array=phase_array)
    
    # =========================================================================
    # Create liver mask
    print("Extracting liver mask...")
    # sum all the 3D phase images and test if > 0
    liv_ind = np.where(np.sum(phase_array, axis = -1) > phase_array.min())
    liver_mask = np.zeros(phase_array.shape[0:3])
    liver_mask[liv_ind] = 1  # a 3D liver mask
    
    #==========================================================================
    # erode liver mask
    print("Eroding liver mask...")
    erosion_param = np.ones((5,5,5))
    liver_mask = ndimage.binary_erosion(liver_mask, erosion_param)
    liv_ind = np.where(liver_mask > 0) # update liver indices
    
    #==========================================================================
    # Bias field correction
    print("Bias field correction:")
    phase_corrected_arr, bias_field_arr = correctBias(phase_array, liver_mask)
    print("\tBias field: min = %0.2f, max = %0.2f" % (np.nanmin(bias_field_arr),
                                                      np.nanmax(bias_field_arr)) )
    np.savez_compressed(dicom_dir + '/phase_corrected_arr',
                        phase_corrected_arr=phase_corrected_arr)
        
    #==========================================================================
    # Create clusters for different Ks (number of clusters)
    clust_sizes = [5,6,7,8,9,10,
                   12,14,16,18,20,
                   23,26,29,32,35,
                   40,45,50,55,60,
                   70,80,90,100]
    
    random.seed(123) # random seed for reproducibility
    clusters = [cluster_phases(phase_array_ = phase_corrected_arr,
                               liv_ind = liv_ind,
                               number_clusters = k) for k in clust_sizes]
    clusters = np.array(clusters) # list to array
    clusters = clusters.transpose(1,2,3,0)
    np.savez_compressed(dicom_dir + '/clusters', clusters=clusters)
    
    print("Clustering pipleine completed succesfully.")

    # LOADING
    # with np.load(dicom_dir + '/clusters.npz') as data:
    #     clusters_x = data['clusters']
    
def load_NPZs(dicom_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    A function that loads dicom files into a 4D phase array.

    Parameters
    ----------
    dicom_dir : str
        A path to directory with all phase subdirectories (with dicom files).
        
    Returns
    ----------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Returns a tuple (original 4d phase array,
                         bias corrected 4d phase array,
                         clusters 3d array)
    """
    with np.load(dicom_dir + '/phase_array.npz') as data:
        phase_array = data['phase_array']
    with np.load(dicom_dir + '/phase_corrected_arr.npz') as data:
        phase_arr_filtered = data['phase_corrected_arr']     
    with np.load(dicom_dir + '/clusters.npz') as data:
        clusters = data['clusters']
    return (
            phase_array,
            phase_arr_filtered,
            clusters
            )
