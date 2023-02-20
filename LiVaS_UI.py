"""
LiVaS User Interface

1) Load preprocessed phase image arrays and clusters
2) Scroll through 2D image slices of a 3D array.
3) Browse through clusters.
4) Increase/decrease number of clusters.
5) Label clusters.
6) Save labels.

Usage: python liVaS_UI.py

License:
    GNU GPL 2.0

Revision: Feb 17, 2023
"""
#%%
import numpy as np
import matplotlib
matplotlib.rcParams['figure.figsize'] = [10, 5] # canvas size
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from LiVaS_helper import *

#%%
class imageSlicer:
    def __init__(self, ax, phase_array, phase_arr_filtered, clusters):
        self.start_time = time.time()
        self.ax = ax
        self.title_text1 = """
        up/down:       Navigate slices
        alt+left/right: Navigate clusters
        alt+up/down: In/De -crease no. clusters
        alt+h/p/t:       Hepatic/Portal/Trash
        q:                   Quit LiVaS UI (or choose next case)"""
        
        self.title_text2 = """
        alt+s: Save labels to disk"""
        
        ax[1].set_title(self.title_text1, fontsize = 9, loc='left')
        ax[0].set_title("Arterial-phase image", fontsize = 10, loc='left')
        ax[2].set_title(self.title_text2, fontsize = 9, loc='left')
        
        self.X = phase_array[:,:,:,1] # ARTEARIAL 3D liver image
        rows, cols, self.slices = self.X.shape
        self.phase_arr_filtered = phase_arr_filtered
        
        #----------------------------------------------------------------------
        # centroid 
        x, y, z = ndimage.center_of_mass(self.X)
        #self.ind = self.slices//2 # slice index
        self.ind = int(round(z)) # centroid slice index
        
        #----------------------------------------------------------------------
        # clusters is a 3D array with integer labels for each cluster
        self.no_cluster_means_idx = 0 
        self.no_cluster_means_max_idx = clusters.shape[-1] - 1
        self.Y = clusters[:,:,:,self.no_cluster_means_idx] # # get a 3d clusters arr
        self.no_cluster_means = self.Y.max() # number of cluster centroids

        self.F = self.phase_arr_filtered[:,:,:,1] # ARTEARIAL 3D filtered image
        print("Filtered phase array initial min = %s, max = %s" % (self.F.min(), self.F.max()) )

        print("3D clusters array shape = %s \n" % str(self.Y.shape))
        self.cl_label = 0 # initialize cluster label to cluster 0
        self.labels = ['Trash'] * (self.no_cluster_means+1) # initialize labels list
                
        #------------------display liver image---------------------------------
        self.im0 = ax[0].imshow(self.X[:, :, self.ind],
                              cmap='gray', alpha = 1, interpolation='none')
        
        #------------------display cluster image-------------------------------
        self.im1a = ax[1].imshow(self.F[:, :, self.ind],
                             cmap='gray', alpha = 1, interpolation='none')
        # self.im1b = ax[1].imshow(self.Y[:, :, self.ind] == self.cl_label,
        #                      cmap='copper', alpha = .5, interpolation='none')
        self.im1b = ax[1].imshow(np.random.randint(2, size=self.Y.shape)[:, :, self.ind],
                             cmap='copper', alpha = .5, interpolation='none')
        
        #------------display filtered liver image and updated clusters---------
        self.im2a = ax[2].imshow(self.F[:, :, self.ind],
                             cmap='gray', alpha = 1, interpolation='none')
        self.Z = np.zeros(self.Y.shape) # initiate updated clusters array
        # initiate random image of integers in [0,1,2] so that the cmap is initialized to 3 colors
        self.im2b = ax[2].imshow(np.random.randint(3, size=self.Y.shape)[:, :, self.ind],
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
        im1a_data = self.im1a.to_rgba(self.F[:, :, self.ind],
                                    alpha=self.im1a.get_alpha())
        im1b_data = self.im1b.to_rgba(self.Y[:, :, self.ind] == self.cl_label,
                                    alpha=self.im1b.get_alpha())
        self.im0.set_data(im0_data) # liver only image
        self.im1a.set_data(im1a_data) # liver
        self.im1b.set_data(im1b_data) # cluster data
        
        im2a_data = self.im2a.to_rgba(self.F[:, :, self.ind], alpha=self.im2a.get_alpha())
        im2b_data = self.im2b.to_rgba(self.Z[:, :, self.ind], alpha=self.im2b.get_alpha())
        self.im2a.set_data(im2a_data)
        self.im2b.set_data(im2b_data)
        ax[2].set_title(self.title_text2, fontsize = 9, loc='left')
        
        self.ax[0].set_xlabel('slice %s' % self.ind)
        self.ax[1].set_xlabel("cluster %s of %s,   Label = %s" %
                           (self.cl_label,
                            self.no_cluster_means,
                            self.labels[self.cl_label]))
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
        
        #----------alt+up/down-------------------------------------------------
        if event.key == 'alt+down':
            self.no_cluster_means_idx = (self.no_cluster_means_idx-1) % 25 # since the highest clust means index is 24
            self.Y = clusters[:,:,:,self.no_cluster_means_idx] # # get a 3d clusters arr
            self.no_cluster_means = self.Y.max() # number of cluster centroids
            #print("3D clusters array shape = %s \n" % str(self.Y.shape))
            self.cl_label = 0 # reset cluster label to cluster 1
            self.labels = ['Trash'] * (self.no_cluster_means+1) # initialize labels list
            self.Z = np.zeros(self.Y.shape) # reset updated clusters array
        if event.key == 'alt+up':
            self.no_cluster_means_idx = (self.no_cluster_means_idx+1) % 25
            self.Y = clusters[:,:,:,self.no_cluster_means_idx] # # get a 3d clusters arr
            self.no_cluster_means = self.Y.max() # number of cluster centroids
            #print("3D clusters array shape = %s \n" % str(self.Y.shape))
            self.cl_label = 0 # reset cluster label to cluster 1
            self.labels = ['Trash'] * (self.no_cluster_means+1) # initialize labels list
            self.Z = np.zeros(self.Y.shape) # reset updated clusters array
         
        #----------alt+left/right----------------------------------------------    
        if event.key == 'alt+left':
            self.cl_label = (self.cl_label - 1) % (self.no_cluster_means+1)
            #print("Cluster label = ", self.cl_label)
            #----------------------------------------------------------------------
            # current cluster label centroid 
            x, y, z = ndimage.center_of_mass(self.Y == self.cl_label)
            #print("centroid z = ", z)
            self.ind = int(round(z)) # centroid slice index
        if event.key == 'alt+right':
            self.cl_label = (self.cl_label + 1) % (self.no_cluster_means+1)
            #print("Cluster label = ", self.cl_label)
            #----------------------------------------------------------------------
            # current cluster label centroid 
            x, y, z = ndimage.center_of_mass(self.Y == self.cl_label)
            #print("centroid z = ", z)
            self.ind = int(round(z)) # centroid slice index
        
        #----------alt+h/p/t---------------------------------------------------    
        if event.key == 'alt+h':
            self.labels[self.cl_label] = 'Hepatic'
            self.Z[self.Y == self.cl_label] = 2
            self.title_text2 = """
            alt+s: Save labels to disk. (NOT SAVED YET!)"""
        if event.key == 'alt+p':
            self.labels[self.cl_label] = 'Portal'
            self.Z[self.Y == self.cl_label] = 1
            self.title_text2 = """
            alt+s: Save labels to disk. (NOT SAVED YET!)"""
        if event.key == 'alt+t':
            self.labels[self.cl_label] = 'Trash'
            self.Z[self.Y == self.cl_label] = 0
            self.title_text2 = """
            alt+s: Save labels to disk. (NOT SAVED YET!)"""
                        
        #----------alt+s-------------------------------------------------------    
        if event.key == 'alt+s':
            self.end_time = time.time()
            with open(dicom_dir + "/time_seconds.txt", mode='w') as file: 
                file.write(str(round(self.end_time - self.start_time)))
            np.savez_compressed(dicom_dir + '/labels', labels=self.Z)
            self.title_text2 = """
            alt+s: Save labels to disk. (SAVED)"""

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
    
    # grab 1st and 2nd -indexed element from the tuple
    phase_arr_filtered, clusters = load_NPZs(dicom_dir)[1:3]

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.canvas.manager.set_window_title("LiVaS UI")
    for j in range(ax.size):
        ax[j].axes.xaxis.set_ticks([])
        ax[j].axes.yaxis.set_ticks([])
    #slicer = imageSlicer(ax, phase_array, phase_arr_filtered, clusters)
    slicer = imageSlicer(ax, phase_arr_filtered, phase_arr_filtered, clusters)
    slicer.connect()
    plt.show()
    
#%%
