import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import skimage
from skimage import io, color
from skimage.filters.rank import entropy
from skimage.morphology import disk, ball
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import img_as_float
from skimage.filters import try_all_threshold, threshold_otsu
from skimage.filters import scharr, sobel, prewitt, roberts
from skimage.feature import canny
from skimage import exposure
import numpy as np


class DTMModel:
    
    def __init__(self, img_name: str):
        
        self.img_dir = 'images/'
        self.script_dir = os.path.dirname(os.path.abspath(self.img_dir))
        self.path = os.path.join(self.script_dir, self.img_dir,img_name)
        self.img = io.imread(self.path)
        
        # Extract filename without extension
        self.filename = os.path.splitext(os.path.basename(self.path))[0]
        
        
        
    def show(self):
        imshow(self.img)
        plt.title(f'Original Image: {self.filename}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        
    def noise_plot(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8),
                       sharex=True, sharey=True)
        
        self.sigma_est = estimate_sigma(self.img, channel_axis=-1, average_sigmas=True)
        # 1. Identify noise type
        if self.img.ndim == 3:
            self.img = color.rgb2gray(self.img)
            
        else:
            self.img = self.img
            
        # Compute the histogram to identify the noise pattern
        hist, bins = np.histogram(self.img.flatten(), bins=256, range=[0,1])
        
        # Plot the histogram
        self.ax[0].grid(True)
        self.ax[0].plot(hist)
        self.ax[0].set_title(f'{self.filename} Noise Pattern')
        self.ax[0].set_xlabel('Pixel Intensity')
        self.ax[0].set_ylabel('Pixel Frequency')
        self.ax[0].legend([f"Sigma: {self.sigma_est:.4f}"])
        self.ax[1].axis('off')
        plt.tight_layout()
        plt.show()
        
        
    def denoise(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8),
                       sharex=True, sharey=True)
        
        self.denoised_img = denoise_tv_chambolle(self.img, channel_axis=-1)
        self.ax[0].imshow(self.img)
        self.ax[0].set_title(f'Original Image of: {self.filename}')
        self.ax[0].axis('off')
        
        self.ax[1].imshow(self.denoised_img)
        self.ax[1].set_title(f'Denoised Image of: {self.filename}')
        self.ax[1].axis('off')
        plt.tight_layout()
        plt.show()
        
        
    def enhance_contrast(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8),
                       sharex=True, sharey=True)
        
        self.original_contrast_img = self.img
        self.enhanced_contrast_img = exposure.equalize_adapthist(self.img, clip_limit=0.1)
        
        self.ax[0].imshow(self.original_contrast_img)
        self.ax[0].set_title('Original Image')
        self.ax[0].axis('off')
        
        self.ax[1].imshow(self.enhanced_contrast_img)
        self.ax[1].set_title('Adaptive Histogram Equalization Applied Image')
        self.ax[1].axis('off')
        plt.tight_layout()
        plt.show()

    
    def contrast_graph(self):
        self.fig, self.ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8),
                       sharex=True, sharey=True)
        
        self.enhanced_contrast_img = exposure.equalize_adapthist(self.img, clip_limit=0.03)
        
        self.ax[0].grid(True)
        self.ax[0].hist(self.img.flatten(),bins=256, color='b')
        self.ax[0].set_title('Histogram of The Original Image')
        self.ax[0].set_xlabel('Pixel Value')
        self.ax[0].set_ylabel('Number of Pixels')
        
        self.ax[1].grid(True)
        self.ax[1].hist(self.enhanced_contrast_img.flatten(),bins=1000, color='b')
        self.ax[1].set_title('Histogram of Contast Enhanced Image')
        self.ax[1].set_xlabel('Pixel Value')
        self.ax[1].set_ylabel('Number of Pixels')
        plt.tight_layout()
        plt.show()

    def thresholds(self):
        self.grayimage = color.rgb2gray(self.img)
        self.entropy_img = entropy(self.grayimage, disk(5))
        
        # self.ax[0,0].imshow(self.entropy_img)
        self.fig, self.ax = try_all_threshold(self.entropy_img, figsize=(12,8), verbose=False)
        plt.show()
        
        
    def feature_extraction(self):
        self.fig, self.ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True, sharey=True)
        self.grayimage = color.rgb2gray(self.img)
        self.entropy_img = entropy(self.grayimage, disk(5))
        
        self.thresh = threshold_otsu(self.entropy_img)
        self.binary_img = self.entropy_img <= self.thresh
        print(self.thresh)
        
        self.ax[0,0].imshow(self.grayimage)
        self.ax[0,0].set_title('Gray scale Image')
        self.ax[0,1].imshow(self.entropy_img)
        self.ax[0,1].set_title(f'Entropy Image (Threshold:{self.thresh:.2f})')
        self.ax[1,0].imshow(self.binary_img)
        self.ax[1,0].set_title('Binary image')
        self.ax[1,1].axis('off')
        
        
        plt.show()

    
    def edge_filter(self):
        
        self.fig, self.ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 8), sharex=True, sharey=True)

        self.grayimage = color.rgb2gray(self.img)
        self.roberts_edge = roberts(self.grayimage)
        self.scharr_edge = scharr(self.grayimage)
        self.prewitt_edge = prewitt(self.grayimage)
        self.sobel_edge = sobel(self.grayimage)
        
        self.ax[0,0].imshow(self.grayimage)
        self.ax[0,0].set_title('Original Image')
        self.ax[0,1].imshow(self.roberts_edge)
        self.ax[0,1].set_title('Roberts Edge')
        self.ax[0,2].imshow(self.scharr_edge)
        self.ax[0,2].set_title('Scharr Edge')
        self.ax[1,0].imshow(self.prewitt_edge)
        self.ax[1,0].set_title('Prewitt Edge')
        self.ax[1,1].imshow(self.sobel_edge)
        self.ax[1,1].set_title('Sobel Edge')
        print(self.ax)
        for axes in self.ax:
            for a in axes:
                a.axis('off')
        plt.show()
        
        return [self.grayimage, self.roberts_edge, self.scharr_edge, self.prewitt_edge, self.sobel_edge]
        
        
    def edge_detector(self):
        # self.fig, self.ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 8), sharex=True, sharey=True)
        self.grayimage = color.rgb2gray(self.img)
        self.edge_canny = canny(self.grayimage, sigma=2)
        
        plt.imshow(self.edge_canny)
        plt.show()
        
        return self.edge_canny
        

# pic = 'r39.jpg'  # Images with low noise: r3, r4
# model = DTMModel(pic)
# model.show()
