# Header
# Author: Dan Ahmad, PhD - For the University of Rochester (UR) - BME Department - TraCe-bMPS
# Version 1.0, split from original code for ML training. January 29th, 2025
# Runs on Python 3.12+

import numpy as np
import os
import sys
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imsave
from skimage.util import img_as_ubyte, img_as_float
from tqdm import tqdm
from scipy.optimize import curve_fit
from av import VideoFrame, container
from aicsimageio import AICSImage


class Raw_Image_Processor:
    '''
    Image Processor class for flattening phase contrast microscopy images.
    Input parameters are image path, desired (square) dimensions, keyframe, and save path.
    The keyframe is the ideal image that we want to match the brightness to, this is provided as a single reference image in the repository.
    Desired_dim is the interpolation parameter, any images larger than this will be downsampled to this size.
    '''
    def __init__(
            self,
            img_path, 
            desired_dim = None, 
            keyframe = None, 
            save_path = None,
            need_video = True,
            vid_fps = 24
                ):
        self.desired_dim = desired_dim
        self.img_path = img_path
        self.grayphaseimlist = []
        self.size_ar = []
        self.im1 = None
        self.ffimg = []
        self.keyframe = keyframe
        self.grayphaseimlist = []
        self.bfimg = []
        self.grayphaseimlist8bit = []
        self.save_path = save_path
        self.xmod_loss = []
        self.need_video = need_video
        self.vid_fps = vid_fps

        # Assertions
        assert self.desired_dim != None, "Desired dimension must be set!"
        assert self.desired_dim > 0, "Desired dimension must be greater than zero!"
        assert self.desired_dim % 2 == 0, "Desired dimension must be even!"
        assert os.path.exists(self.img_path), "Image path does not exist!"
        assert os.path.exists(self.keyframe), "Keyframe path does not exist!"
        assert os.path.exists(self.save_path), "Save path does not exist!"

        # TO DO: Set directories to default to current working directory.

    # Image reading fcn, reads from "img_path" and stores the image in "im1".
    def read_image(self):        
        filesplit = self.img_path.rsplit('.', 1)

        if len(filesplit) > 1:
            extension = filesplit[1]
        else:
            print('Extension not found, check file naming convention!')

        if extension == 'nd2':
            # Load AICS 
            img = AICSImage(self.img_path)
            self.im1 = img.get_image_data("TYX")
        elif extension == 'tif' or extension == 'tiff':
            # Load TIFF
            self.im1 = io.imread(self.img_path)
        else:
            print('Unsupported file format!')

        self.keyframe = img_as_float(io.imread(self.keyframe))
        if self.desired_dim == None:
            self.desired_dim = self.im1.shape[0]
    
    def convert_to_grayscale(self):
        '''
        Convert RGB to Grayscale, if the image is already grayscale it will skip this step.
        Note that this step is very memory intensive, typical microscope images shoud be grayscale already so this hopefully won't be needed.
        '''
        if len(self.im1[0].shape) == 3:
            for i in tqdm(range(len(self.im1)), desc='Converting RGB to Grayscale'):
                self.grayphaseimlist.append(rgb2gray(self.im1[i]))
        else:
            for i in range(len(self.im1)):
                self.grayphaseimlist.append(self.im1[i])
            print("Grayscale images detected, no need to convert.")
    
    # Collect the size of each image in the stack. Used later in the resize_images function.
    def collect_image_size(self):
        for i in range(len(self.grayphaseimlist)):
            self.size_ar.append(self.grayphaseimlist[i].shape)
    
    # Currently only works for square images, working on adding conditionals for rectangular images.
    #TO DO: Add non-square image processing. 
    def resize_images(self):
        message_flag_check = False
        if all(x == self.size_ar[0] for x in self.size_ar):
            print('All images are the same size')
            print('Checking for downsampling, target is: (' + str(self.desired_dim) + 'x' + str(self.desired_dim) + ')')
            for i in tqdm(range(len(self.size_ar)), desc = 'Downsampling images'):
                if self.size_ar[i][0] > self.desired_dim and self.size_ar[i][1] > self.desired_dim:
                    self.grayphaseimlist[i] = resize(self.grayphaseimlist[i], (self.desired_dim, self.desired_dim), order=3, preserve_range=True)   
                else:
                    if not message_flag_check:
                        print('Images are already (' + str(self.desired_dim) + 'x' + str(self.desired_dim) + ') or smaller')
                        message_flag_check = True
        else: # Right now it doesn't make sense to accept image stacks with different sizes, I don't even think this is a real use case.
            print('Images are not the same size')
            print('It looks like frame', self.size_ar.index(self.size_ar)+1, 'is a different size')
            print('Please check the image stack and try again')
            sys.exit()
    
    def divideflatfield(self, input_img):
        '''
        Flatfield correction, minimially invasive correction, shouldn't mess things up.
        Note, brings the histogram range to 0-4 ish, which is a weird arbitrary range but we clean it up with min-max scaling later.
        This function is kept independent to keep it more modular, due to this it also returns a value.
        '''
        grayscale_img = input_img
        poly2d_fcn = lambda xy, a, b, c, d, e, f: a + b*xy[0] + c*xy[1] + d*xy[0]*xy[0] + e*xy[1]*xy[1] + f*xy[0]*xy[1]

        y, x = np.indices(grayscale_img.shape)

        x_co = x.flatten()
        y_co = y.flatten()
        pix_val = grayscale_img.flatten()
        # Initial guess, works well for all tested images/videos.
        # For now a 2nd order polynomial works well, no need to go higher.
        p0 = [1, 1, 1, 1, 1, 1]
        popt, _ = curve_fit(poly2d_fcn, (x_co, y_co), pix_val, p0=p0) 
        flat_field_img = poly2d_fcn((x_co, y_co), *popt).reshape(grayscale_img.shape)
        fit_img = grayscale_img / (flat_field_img + 1e-6)

        return fit_img
    
    def runflatfield(self):
        for i in tqdm(range(len(self.grayphaseimlist)), desc='Removing flatfield'):
            self.ffimg.append(self.divideflatfield(self.grayphaseimlist[i]))
    
    def means_match(self):
        kfmod = self.divideflatfield(self.keyframe)
        kfmean = np.mean(kfmod)

        self.bfimg = []
        self.xmod_loss = []

        for i in range(len(self.ffimg)):
            self.xmod_loss.append([])

        for i in tqdm(range(len(self.ffimg)), desc='Adjusting Contrast'):
            best_mean_diff = np.inf
            best_xmod = 0
            mean_diff = 0
            bftest = self.ffimg[i]
            bfmin = np.min(bftest)
            
            for xmod in np.linspace(0.1, 10, 200):
                xmodtest = np.clip(xmod * (bftest - bfmin), np.min(kfmod), np.max(kfmod))
                mean_xmodtest = np.mean(xmodtest)
                mean_diff = abs(mean_xmodtest - kfmean)

                self.xmod_loss[i].append(mean_diff) # Diagnostic
                
                if mean_diff < best_mean_diff:
                    best_mean_diff = mean_diff
                    best_xmod = xmod

                if mean_diff < 0.0005:
                    tqdm.write(f'Frame {i+1} is at an acceptable target, stopping iterations')
                    break

            self.bfimg.append(np.clip(best_xmod * (self.ffimg[i] - np.min(self.ffimg[i])), np.min(kfmod), np.max(kfmod)))
    
    def brightness_eq(self, img_inp, btarget):
        bcurrent = np.mean(img_inp)
        scalef = btarget / bcurrent
        return img_inp * scalef
    
    def ubyte_convert(self):
        for i in tqdm(range(len(self.bfimg)), desc='Converting to 8-bit and Brightness Matching'):
            if self.bfimg[i].dtype == 'float64' or self.bfimg[i].dtype == 'uint16' or self.bfimg[i].dtype == 'float32':
                if i == 0:
                    normalized_img = (self.bfimg[i] - np.min(self.bfimg[i])) / (np.max(self.bfimg[i]) - np.min(self.bfimg[i]))
                    fzero_brightness = np.mean(normalized_img)
                    self.grayphaseimlist8bit.append(img_as_ubyte(normalized_img))
                else:
                    normalized_img = (self.bfimg[i] - np.min(self.bfimg[i])) / (np.max(self.bfimg[i]) - np.min(self.bfimg[i]))
                    beq = self.brightness_eq(normalized_img, fzero_brightness)
                    np.clip(beq, 0, 1, out=beq)
                    self.grayphaseimlist8bit.append(img_as_ubyte(beq))
            elif self.bfimg[i].dtype == 'uint8':
                self.grayphaseimlist8bit.append(self.bfimg[i])
                if not message_flag:
                    print('Images are already 8-bit')
                    message_flag = True
            else:
                if not message_flag:
                    print('Images are in an unsupported format, please check the image stack and try again')
                    message_flag = True
                    break
    
    def save_fcn(self):
        print(f'Saving images to: {self.save_path}')
        if not os.path.exists(self.save_path):
            os.makedirs
        file_name = self.img_path.split('/')[-1]
        file_name = file_name.split('.')[0]
        imsave(self.save_path + file_name + '_mod.tif', np.array(self.grayphaseimlist8bit))
        print('A .tif file has been saved successfully')

    def compressed_video(self):
        file_name = self.img_path.split('/')[-1]
        file_name = file_name.split('.')[0]

        with container.open(self.save_path + file_name + '_vid.mp4', mode='w', format='mp4') as output_container:
            stream = output_container.add_stream('h264', self.vid_fps)
            if self.grayphaseimlist8bit:
                sample_image = self.grayphaseimlist8bit[0]
                stream.width = sample_image.shape[1] 
                stream.height = sample_image.shape[0]
                
            stream.options = {'crf': '24'} 
            frames = []
            for image in self.grayphaseimlist8bit:
                frame = VideoFrame.from_ndarray(image, format='gray8')
                frames.append(frame)

            for frame in frames:
                for packet in stream.encode(frame):
                    output_container.mux(packet)
            
            for packet in stream.encode():
                output_container.mux(packet)

        print(f"Video saved as {file_name}_vid.mp4")


    def run_beq(self):
        self.read_image()
        self.convert_to_grayscale()
        self.collect_image_size()
        self.resize_images()
        self.runflatfield()
        self.means_match()
        self.ubyte_convert()
        self.save_fcn()
        if self.need_video:
            self.compressed_video()
        else:
            print("No video saved")
        print('All tasks completed successfully')

    def run_all_no_resize(self):
        self.read_image()
        self.convert_to_grayscale()
        self.runflatfield()
        self.means_match()
        self.ubyte_convert()
        self.save_fcn()

    def xmod_loss_plot(self):
        plt.figure()
        for i in range(len(self.xmod_loss)):
            plt.plot(self.xmod_loss[i], label=f'Image {i+1}')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Difference')
        plt.title('Xmod Loss')
        plt.savefig(self.save_path + 'xmod_loss_plot.png')
        plt.close()