"""
Liver Label Viewer

1) Load preprocessed phase image arrays and LABELS.
2) Scroll through 2D image slices of a 3D array.

Usage: python liverLabelViewer.py

Authors:
    Mladen Zecevic, mladenze@hotmail.com
    Kyle Hasenstab, kylehasenstab@gmail.com
    
Date: Jul 12, 2022
"""
#%%
import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = [10, 5] # canvas size
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog#, messagebox
import scipy.ndimage as ndimage
#import random
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from LiVaS_helper import *

def load_NPZs(dicom_dir):
    #--------------------------------------------------------------------------
    # LOADING saved data
    with np.load(dicom_dir + '/phase_array.npz') as data:
        phase_array = data['phase_array']
    # with np.load(dicom_dir + '/phase_corrected_arr.npz') as data:
    #     phase_arr_filtered = data['phase_corrected_arr']
        
    with np.load(dicom_dir + '/labels.npz') as data:
        labels = data['labels']
    return (
            phase_array,
            #phase_arr_filtered,
            labels
            )

#%%
class labelViewer:
    def __init__(self, ax, phase_array, phase_arr_filtered, labels):
        self.start_time = time.time()
        self.ax = ax
        self.title_text1 = """
        Hepatic (Yellow) and Portal (Pink) veins
        up/down:   navigate slices
        q:               quit slicer (or choose next case)"""
        
# =============================================================================
#         self.title_text2 = """
#         alt+s: Save labels to disk"""
# =============================================================================
        
# =============================================================================
#         ax[1].set_title(self.title_text1, fontsize = 9, loc='left')
# =============================================================================
        ax[0].set_title("Arterial phase image (pre- bias field correction)", fontsize = 10, loc='left')
        ax[1].set_title(self.title_text1, fontsize = 10, loc='left')
        
        self.X = phase_array[:,:,:,1] # ARTEARIAL 3D liver image
        rows, cols, self.slices = self.X.shape
        self.phase_arr_filtered = phase_arr_filtered
        
        #----------------------------------------------------------------------
        # centroid 
        x, y, z = ndimage.center_of_mass(self.X)
        #self.ind = self.slices//2 # slice index
        self.ind = int(round(z)) # centroid slice index
        

        self.F = self.phase_arr_filtered[:,:,:,1] # ARTEARIAL 3D filtered image
        print("Filtered phase array initial min = %s, max = %s" % (self.F.min(), self.F.max()) )

                
        #------------------display liver image---------------------------------
        self.im0 = ax[0].imshow(self.X[:, :, self.ind],
                              cmap='gray', alpha = 1, interpolation='none')
        
        #------------display filtered liver image and updated clusters---------
        self.im1a = ax[1].imshow(self.F[:, :, self.ind],
                             cmap='gray', alpha = 1, interpolation='none')
        self.Z = labels
        # initiate random image of integers in [0,1,2] so that the cmap is initialized to 3 colors
        self.im1b = ax[1].imshow(self.Z[:, :, self.ind],
                             cmap='inferno', alpha = .7, interpolation='none')
        
        plt.tight_layout(pad=0) # remove figure padding
        self.update()
        
    def connect(self):
        #----------------------------------------------------------------------
        # Connect to all the needed events.
        self.cidpress = self.im1a.axes.figure.canvas.mpl_connect(
            'key_press_event', self.on_press)
        self.cidscroll = self.im1a.axes.figure.canvas.mpl_connect(
            'scroll_event', self.on_scroll)
        
    def disconnect(self):
        #----------------------------------------------------------------------
        # Disconnect all callbacks.
        self.im1a.axes.figure.canvas.mpl_disconnect(self.cidpress)
        self.im1a.axes.figure.canvas.mpl_disconnect(self.cidscroll)

    def update(self):
        im0_data = self.im0.to_rgba(self.X[:, :, self.ind],
                                    alpha=self.im0.get_alpha())
        self.im0.set_data(im0_data) # liver only image
        
        im1a_data = self.im1a.to_rgba(self.F[:, :, self.ind], alpha=self.im1a.get_alpha())
        im1b_data = self.im1b.to_rgba(self.Z[:, :, self.ind], alpha=self.im1b.get_alpha())
        self.im1a.set_data(im1a_data)
        self.im1b.set_data(im1b_data)
        #ax[1].set_title(self.title_text1, fontsize = 9, loc='left')
        
        self.ax[0].set_xlabel('slice %s' % self.ind)
        self.im1a.axes.figure.canvas.draw()
        self.im1b.axes.figure.canvas.draw()
        
        #self.im2a.axes.figure.canvas.draw()
        #self.im2b.axes.figure.canvas.draw()
        #self.fig.canvas.flush_events()
        
    def on_scroll(self, event):
        #print("scroll %s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()
        
    def on_press(self, event):
        #print("press %s" % event.key)
        
        #----------up/down-----------------------------------------------------
        if event.key == 'down':
            self.ind = (self.ind - 1) % self.slices
        if event.key == 'up':
            self.ind = (self.ind + 1) % self.slices
        self.update()

#%%
# =============================================================================
# The Meat
# =============================================================================
while (True):
    dicom_dir = folderChooser()
    if dicom_dir == 'break': break # exit the while loop
    print("DICOM DIR = %s" % dicom_dir)
    # phase_array, phase_arr_filtered, clusters = load_NPZs(dicom_dir)
    phase_array, labels = load_NPZs(dicom_dir)

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.canvas.manager.set_window_title("LiVaS Label Viewer")
    for j in range(ax.size):
        ax[j].axes.xaxis.set_ticks([])
        ax[j].axes.yaxis.set_ticks([])
    viewer = labelViewer(ax, phase_array, phase_array, labels)
    #slicer = imageSlicer(ax, phase_arr_filtered, phase_arr_filtered, labels)
    viewer.connect()
    plt.show()
    
#%%
