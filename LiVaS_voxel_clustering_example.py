"""
LiVaS_voxel_clustering_example

An example on how to run voxel clustering pipeline. 

Usage: LiVaS_voxel_clustering_example "/path/to/a/case/to/be/processed/"
    The supplied path's target directory should contain subdirectories with 
    phase dicom files.'
    
License:
    GNU GPL 2.0
    
Revision: Feb 19, 2023
"""
#%%
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from LiVaS_helper import *

#%%

if len(sys.argv) > 2:
    print( "Usage: python LiVaS_voxel_clustering_example.py '/path/to/case/directory'" )
    sys.exit(1)
elif len(sys.argv) == 2:
    dicom_dir = os.path.normpath(str(sys.argv[1]))
    if not os.path.exists(dicom_dir):
        print("The provided path does not exist.")
elif len(sys.argv) < 2:
    dicom_dir = folderChooser()
        
#------------------------------------------------------------------------------
# List containing search kewords to be used in sorting phase series.
# The order of search keys in the list should reflect the AP timing,
#  and should contain substrings of the relevant phase-directories' names.
sort_order_key_list = ["PRE","ARTERIAL", "PVP.M", "PVP", "DELAYED.M",
                       "DELAYED.","HBP.M","HBP.","aDummy"]

# run the clustering pipeline
voxel_clustering_pipeline(dicom_dir, sort_order_key_list)

