import os 
import cv2
import numpy as np

from numba import njit
import time 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

DITHER_BRIGHTNESS = 32

# define global variables 
BAYER_KERNEL = {2: np.array([[0, 2], 
                             [3, 1]]) * 1/4,
                  4: np.array([[0, 8, 2, 10], 
                               [12, 4, 14, 6],
                               [3, 11, 1, 9],
                               [15, 7, 13, 5]]) * 1/16,
                  8: np.array([[ 0, 32,  8, 40,  2, 34, 10, 42], 
                               [48, 16, 56, 24, 50, 18, 58, 26],
                               [12, 44,  4, 36, 14, 46,  6, 38],
                               [60, 28, 52, 20, 62, 30, 54, 22],
                               [ 3, 35, 11, 43,  1, 33,  9, 41],
                               [51, 19, 59, 27, 39, 17, 57, 25],
                               [15, 47,  7, 39, 13, 45,  5, 37],
                               [63, 31, 55, 23, 61, 29, 53, 21]])*1/64,
                  }

GX = {0: np.array([[1, 0, -1],          # Sobel
                   [2, 0, -2],
                   [1, 0, -1]]),
      1: np.array([[1, 0, -1],          # Prewitt
                   [1, 0, -1],
                   [1, 0, -1]]),
      2: np.array([[3, 0, -3],          # Scharr
                   [10, 0, -10],
                   [3, 0, -3]]),
      }
GY = {0: np.array([[1, 2, 1],
                   [0, 0, 0],
                   [1, -2, -1]]),
      1: np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]]),
      2: np.array([[3, 10, 3],
                   [0, 0, 0],
                   [-3, -10, -3]]),
      }

GAUSSIAN_FILTER = np.array([[  2/159,  4/159,  5/159,  4/159,  2/159], 
                            [  4/159,  9/159, 12/159,  9/159,  4/159], 
                            [  5/159, 12/159, 15/159, 12/159,  5/159], 
                            [  4/159,  9/159, 12/159,  9/159,  4/159], 
                            [  2/159,  4/159,  5/159,  4/159,  2/159]])


DITHER_ARRAY = {0: np.array([[0,        0,       0,     7/16,       0],              # floyd steinberg
                             [0,     3/16,    5/16,     1/16,       0],
                             [0,        0,       0,        0,       0]]),
                1: np.array([[0,        0,       0,      3/8,       0],               # false floyd steinberg
                             [0,        0,     3/8,      2/8,       0],
                             [0,        0,       0,        0,       0]]),
                2: np.array([[0,        0,       0,     7/48,    5/48],               # jarvis, judice, and ninke
                             [3/48,  5/48,    7/48,     4/48,    3/48],
                             [1/48,  3/48,    5/48,     3/48,    1/48]]),
                3: np.array([[0,        0,       0,     8/42,    4/42],               # stucki
                             [2/42,  4/42,    8/42,     4/42,    2/42],
                             [1/42,  2/42,    4/42,     3/42,    1/42]]),                       
                4: np.array([[0,        0,       0,      1/8,     1/8],               # Atkinson
                             [0,      1/8,     1/8,      1/8,       0],
                             [0,        0,     1/8,        0,       0]]),    
                5: np.array([[0,        0,       0,     8/32,    4/32],               # Burkes 
                             [2/32,  4/32,    8/32,     4/32,    2/32],
                             [0,        0,       0,        0,       0]]),      
                6: np.array([[0,        0,       0,     5/32,    3/32],               # Sierra
                              [2/32, 4/32,    5/32,     4/32,    2/32],
                              [0,    2/32,    3/32,     2/32,       0]]),                       
                7: np.array([[0,        0,       0,     4/16,    3/16],               # Two-row Sierra
                              [1/16, 2/16,    3/16,     2/16,    1/16],
                              [0,       0,       0,        0,       0]]),    
                8: np.array([[0,        0,       0,      2/4,       0],               # Sierra Lite 
                              [0,     1/4,     1/4,        0,       0],
                              [0,       0,       0,        0,       0]]),    
                }

DITHER_NAMES = {0: 'Floyd Steinberg',
                1: 'False Floyd Steinberg',
                2: 'Jarvis, Judice, Ninke',
                3: 'Stucki',
                4: 'Atkinson', 
                5: 'Burkes', 
                6: 'Sierra', 
                7: 'Two-Row Sierra',
                8: 'Sierra Lite'}

SHARPENING_KERNELS = {0: np.array([[ 0, -1,  0],
                                   [-1,  5, -1],
                                   [ 0, -1,  0]]),
                      1: np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]]),
                      }

IGN_PRESETS = {0: (52.9829189, 0.06711056, 0.00583715),
               1: (49.7842291094547,0.606643128024062, 0.171439780306002 ),}

# https://observablehq.com/@jobleonard/pseudo-blue-noise, link to generate more IGN prests 


# 0.0 get current working directory and check if inputs and outputs exist
PATH = os.getcwd()

main_paths = {'Inputs': 'inputs', 'Outputs': 'outputs', 'Color Schemes': 'color_schemes', 'Resources': 'resources'}

for key in main_paths:
    if not os.path.exists(main_paths[key]):
        # Create the folder
        os.makedirs(main_paths[key])
    else:
        pass
    main_paths[key] = os.path.join(PATH, main_paths[key])
   

# 0.1 Import Color schemes if needed
def color_scheme_import(palette_size=32):
    color_palette_images = [f for f in os.listdir(main_paths['Color Schemes']) if os.path.isfile(os.path.join(main_paths['Color Schemes'], f))]
    color_schemes = {}
    for num, col_pal_file in enumerate(color_palette_images):
        col_pal_array = cv2.imread(os.path.join(main_paths['Color Schemes'], col_pal_file))
        col_pal_array = np.flip(col_pal_array, axis=-1) 
        colors = {}
        for x in range(col_pal_array.shape[1] // palette_size):
            # get first pixel of each cube in color palette
            colors[x] = tuple(col_pal_array[0, x*palette_size, :].flatten())
            
        color_schemes[num] = {'Name': col_pal_file, 'colors': colors}          
    return color_schemes   

color_schemes = color_scheme_import()


# 0.2 Function to load image and color scheme 
def load_image_and_colors(input_name, colors, downscale=1, rescale=1, other_array=None):
    # input_name:   filename of file in inputs folder 
    # colors:       color scheme in dict format  
    # downscale:    scale to which the image shall be scaled
    # rescale:      scale to which the image shall be scaled in the end
    # other_array:  to chain render pipeline
    # final_dims:   target dims of final output 
    
    if other_array is None:
        img_array = cv2.imread(os.path.join(main_paths['Inputs'], input_name))
    else:
        img_array = other_array
        
    final_dims = ((img_array.shape[1] // rescale), (img_array.shape[0] // rescale))
    input_dims = ((img_array.shape[1] // downscale), (img_array.shape[0] // downscale))
    img_array = cv2.resize(img_array, input_dims, interpolation=cv2.INTER_CUBIC)
    img_array = img_array.astype('float64')    
    img_array = np.flip(img_array, axis=-1)         
        
    # get colors from color dictionary
    if colors != None:
        color_array = np.asarray(list(colors.values()))
    else:
        color_array = None
    return img_array, color_array, final_dims


# Import blue noise textures 
BLUE_NOISE = {}
for kernel_size in [16, 32, 64, 128]:
    input_name = 'blue_noise_' + str(kernel_size) + '.png'
    noise_texture = cv2.imread(os.path.join(main_paths['Resources'], input_name), 0)
    BLUE_NOISE[kernel_size] = np.array(noise_texture) / 255


# --------------------------------------------------------------------------- #
# =====================             DITHERING             =================== #
# --------------------------------------------------------------------------- #
# 1.1 bayer dithering algorithm
@njit
def bayer_algo(img_array, kernel_size, output_array, bayer_array, color_array):
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            oldpixel = img_array[y, x]
            oldpixel += DITHER_BRIGHTNESS  * bayer_array[x % kernel_size, y % kernel_size]
            x_pixel = np.power(color_array - oldpixel, 2)
            x_pixel = np.sum(x_pixel, axis=1)
            newpixel = color_array[np.argmin(x_pixel)]
            output_array[y, x] += newpixel   
    return output_array

# wrapper for the actual algorithm
def bayer_dither(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, kernel_size=2, render=False):
    
    # Load image
    img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array)
    
    # output array 
    output_array = np.zeros_like(img_array)
    bayer_array = BAYER_KERNEL[kernel_size]
    
    # Loop through pixels and apply bayer lattice
    output_array = bayer_algo(img_array, kernel_size, output_array, bayer_array, color_array,)
    
    # rendering
    if render:
        cv2.imshow('Bayer Dither', np.flip(output_array/256, axis=-1))
        cv2.waitKey(0)

    output_array = np.flip(output_array, axis=-1) 
    opencvimg = cv2.resize(output_array, final_dims, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(main_paths['Outputs'], output_name), opencvimg)
    return output_array


# --------------------------------------------------------------------------- #
# 1.2 Floyd-steinberg dithering algorithm
@njit
def dither_algo(output_array, color_array, dither_array):
    # Loop through pixels and apply floyd steinberg

    
    for y in range(0, output_array.shape[0]-2):
        for x in range(2, output_array.shape[1]-2):
            oldpixel = np.array(list(output_array[y, x]))
            
            x_pixel = np.power(color_array - oldpixel, 2)
            x_pixel = np.sum(x_pixel, axis=1)
            newpixel = color_array[np.argmin(x_pixel)]
            
            output_array[y, x] = newpixel
            quant_error = oldpixel - newpixel
            
            for i in range(dither_array.shape[0]):
                for j in range(dither_array.shape[1]):
                    if dither_array[i, j] != 0:
                        add_error = output_array[y+i, x-2+j] + quant_error * dither_array[i, j]
                        if add_error.sum() > 765:
                            output_array[y+i, x-2+j] = np.array([255, 255, 255]).astype('float64')  
                        elif add_error.sum() <0:
                            output_array[y+i, x-2+j] = np.array([0, 0, 0]).astype('float64')  
                        else:
                            output_array[y+i, x-2+j] += quant_error * dither_array[i, j]
                    
    return output_array


# wrapper for the dithering process
def dithering(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, algorithm=0):
    
    # Load image
    img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array)
    
    # output array 
    output_array = np.zeros((img_array.shape[0]+ 2, img_array.shape[1]+ 4, img_array.shape[2]))
    output_array[0:-2, 2:-2, :] += img_array
    
    # Implementation of the actual dithering
    output_array = dither_algo(output_array, color_array, DITHER_ARRAY[algorithm])
    output_array = output_array[0:-2, 2:-2, :]
    
    if render:
        cv2.imshow(DITHER_NAMES[algorithm], np.flip(output_array/256, axis=-1))
        cv2.waitKey(0)

    output_array = np.flip(output_array, axis=-1) 
    opencvimg = cv2.resize(output_array, final_dims, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(main_paths['Outputs'], output_name), opencvimg)
    return output_array

# --------------------------------------------------------------------------- #
# 1.3 Interleaved Gradient Noise Dithering
@njit 
def ign_algo(img_array, color_array, output_array, ign_preset):
    for y in range(0, img_array.shape[0]):
        for x in range(0, img_array.shape[1]):
            oldpixel = img_array[y, x]
            v = ign_preset[0] * (ign_preset[1] * x + ign_preset[2] * y)
            v = v - np.floor(v)
            oldpixel += DITHER_BRIGHTNESS * v
            x_pixel = np.power(color_array - oldpixel, 2)
            x_pixel = np.sum(x_pixel, axis=1)
            output_array[y, x] = color_array[np.argmin(x_pixel)]

    return output_array

def ign_dither(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, ign_preset=0):
    # Load image
    img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array)
    
    # output array 
    output_array = np.zeros_like(img_array)
    ign_preset = IGN_PRESETS[ign_preset]
    # Run color substitution algorithm            
    output_array = ign_algo(img_array, color_array, output_array, ign_preset)
    
    # rendering
    if render:
        cv2.imshow('Color Palette Swap', np.flip(output_array/256, axis=-1))
        cv2.waitKey(0)


    output_array = np.flip(output_array, axis=-1) 
    opencvimg = cv2.resize(output_array, final_dims, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(main_paths['Outputs'], output_name), opencvimg)
    return output_array


# --------------------------------------------------------------------------- #
# 1.4 Blue Noise Dithering
@njit 
def blue_noise_algo(img_array, kernel_size, output_array, blue_noise, color_array):
    for y in range(img_array.shape[0]):
        for x in range(img_array.shape[1]):
            oldpixel = img_array[y, x]
            oldpixel += DITHER_BRIGHTNESS  * blue_noise[x % kernel_size, y % kernel_size]
            x_pixel = np.power(color_array - oldpixel, 2)
            x_pixel = np.sum(x_pixel, axis=1)
            newpixel = color_array[np.argmin(x_pixel)]
            output_array[y, x] += newpixel   
    return output_array

def blue_noise_dither(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, kernel_size=64):
    # Load image
    img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array)
    
    # output array 
    output_array = np.zeros_like(img_array)
    blue_noise = BLUE_NOISE[kernel_size]
    
    
    # Loop through pixels and apply bayer lattice
    output_array = blue_noise_algo(img_array, kernel_size, output_array, blue_noise, color_array,)
    
    # rendering
    if render:
        cv2.imshow('Bayer Dither', np.flip(output_array/256, axis=-1))
        cv2.waitKey(0)

    output_array = np.flip(output_array, axis=-1) 
    opencvimg = cv2.resize(output_array, final_dims, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(main_paths['Outputs'], output_name), opencvimg)
    return output_array

# --------------------------------------------------------------------------- #
# =====================        OUTLINE ADDITION           =================== #
# --------------------------------------------------------------------------- #
# 2.0 Sobel Edge Detection
@njit
def edge_algo(output_array, img_array, color_array, gamma, gx, gy):
    threshold = 127*3
    min_color = color_array[np.argmin(np.sum(color_array, axis=1))]
    # Loop through pixels and apply bayer lattice
    for y in range(2, img_array.shape[1]-2):
        for x in range(2, img_array.shape[0]-2):
            new_pixel = np.zeros(3)
            for rgb in range(0, 3):
                a = np.sum(gx * img_array[x-1:x+2, y-1:y+2, rgb]) * gamma
                b = np.sum(gy * img_array[x-1:x+2, y-1:y+2, rgb]) * gamma
                new_pixel[rgb] = np.sqrt(a**2 + b**2)                       
            if new_pixel.sum() >= threshold:
                output_array[x, y] = min_color
    return output_array
    

# wrapper for the sharpening algorithm
def edge_sharpen(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, algorithm=0, gamma=1):
    
    # Load image
    img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array)
    
    # output array 
    output_array = np.zeros((img_array.shape[0]+ 4, img_array.shape[1]+ 4, img_array.shape[2]))
    output_array[2:-2, 2:-2, :] += img_array
    
    input_array = np.zeros_like(output_array)
    input_array[2:-2, 2:-2, :] += img_array
    
    # Implementation of the actual dithering
    gx = GX[algorithm]
    gy = GY[algorithm]
    output_array = edge_algo(output_array, input_array, color_array, gamma, gx, gy)

    
    output_array = output_array[2:-2, 2:-2, :] 
    
    if render:
        cv2.imshow('Sobel', np.flip(output_array/256, axis=-1))
        cv2.waitKey(0)

    output_array = np.flip(output_array, axis=-1) 
    opencvimg = cv2.resize(output_array, final_dims, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(main_paths['Outputs'], output_name), opencvimg)
    return output_array


# --------------------------------------------------------------------------- #
# # 2.1 Canny Edge Detection
# @njit
# def canny_algo(output_array, img_array, color_array, gamma, gx, gy):
#     threshold = 127*3
#     min_color = color_array[np.argmin(np.sum(color_array, axis=1))]
#     blurred_image = np.zeros_like(output_array)
#     angles = np.zeros_like(output_array)
    
#     # Apply gaussian filter 
#     for y in range(3, img_array.shape[1]-3):
#         for x in range(3, img_array.shape[0]-3):
#             new_pixel = np.zeros(3)
#             for rgb in range(0, 3):
#                 a = np.sum(GAUSSIAN_FILTER * img_array[x-2:x+3, y-2:y+3, rgb])
#                 new_pixel[rgb] = a      
#             x_pixel = np.power(color_array - new_pixel, 2)
#             x_pixel = np.sum(x_pixel, axis=1)
#             blurred_image[x, y] = color_array[np.argmin(x_pixel)]

#     # Loop through pixels and apply bayer lattice
#     for y in range(2, img_array.shape[1]-2):
#         for x in range(2, img_array.shape[0]-2):
#             new_pixel = np.zeros(3)
#             angle_data = np.zeros(3)
#             for rgb in range(0, 3):
#                 a = np.sum(gx * img_array[x-1:x+2, y-1:y+2, rgb]) * gamma
#                 b = np.sum(gy * img_array[x-1:x+2, y-1:y+2, rgb]) * gamma
#                 new_pixel[rgb] = np.sqrt(a**2 + b**2)   
#                 angle_data = a/b 
#             angles = np.arctan(np.mean(angle_data))
#             if new_pixel.sum() >= threshold:
#                 output_array[x, y] = min_color
#     return output_array
    

# # wrapper for the sharpening algorithm
# def canny_edge_detection(input_name, output_name, colors, downscale=4, rescale=1, other_array=None, final_dims=(0,0), render=False, algorithm=0, gamma=1):
    
#     # Load image
#     img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array, final_dims=final_dims)
    
#     # output array 
#     output_array = np.zeros((img_array.shape[0]+ 6, img_array.shape[1]+ 6, img_array.shape[2]))
#     output_array[3:-3, 3:-3, :] += img_array
    
#     input_array = np.zeros_like(output_array)
#     input_array[3:-3, 3:-3, :] += img_array
    
#     # Implementation of the actual dithering
#     gx = GX[algorithm]
#     gy = GY[algorithm]
#     output_array = canny_algo(output_array, input_array, color_array, gamma, gx, gy)

    
#     output_array = output_array[3:-3, 3:-3, :] 
    
#     if render:
#         cv2.imshow('Sobel', np.flip(output_array/256, axis=-1))
#         cv2.waitKey(0)

#     output_array = np.flip(output_array, axis=-1) 
#     opencvimg = cv2.resize(output_array, final_dims, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
#     cv2.imwrite(os.path.join(main_paths['Outputs'], output_name), opencvimg)
#     return output_array

# --------------------------------------------------------------------------- #
# =====================        IMAGE SHARPENING           =================== #
# --------------------------------------------------------------------------- #

# Sharpening algorithm
@njit
def sharpen_algo(output_array, img_array, color_array, kernel, gamma):

    
    # Loop through pixels and apply bayer lattice
    for y in range(2, img_array.shape[1]-2):
        for x in range(2, img_array.shape[0]-2):
            new_pixel = np.zeros(3)
            for rgb in range(0, 3):
                a = np.sum(kernel * img_array[x-1:x+2, y-1:y+2, rgb]) * gamma
                new_pixel[rgb] = a      
            x_pixel = np.power(color_array - new_pixel, 2)
            x_pixel = np.sum(x_pixel, axis=1)
            newpixel = color_array[np.argmin(x_pixel)]
            output_array[x, y] = newpixel   
    return output_array

# wrapper for the sharpening algorithm
def image_sharpen(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, gamma=1, algorithm=0):
    
    # Load image
    img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array)
    
    # output array 
    output_array = np.zeros((img_array.shape[0]+ 4, img_array.shape[1]+ 4, img_array.shape[2]))
    
    input_array = np.zeros_like(output_array)
    input_array[2:-2, 2:-2, :] += img_array
    
    # Implementation of the actual dithering
    output_array = sharpen_algo(output_array, input_array, color_array, SHARPENING_KERNELS[algorithm], gamma)

    
    output_array = output_array[2:-2, 2:-2, :] 
    
    if render:
        cv2.imshow('Sharpen', np.flip(output_array/256, axis=-1))
        cv2.waitKey(0)

    output_array = np.flip(output_array, axis=-1) 
    opencvimg = cv2.resize(output_array, final_dims, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(main_paths['Outputs'], output_name), opencvimg)
    return output_array

# --------------------------------------------------------------------------- #
# =====================          OTHER TOOLS              =================== #
# --------------------------------------------------------------------------- #
# 3.1 Linear color swap
@njit 
def swap_algo(img_array, color_array, output_array):
    for y in range(0, img_array.shape[1]):
        for x in range(0, img_array.shape[0]):
            oldpixel = img_array[x, y]
            
            x_pixel = np.power(color_array - oldpixel, 2)
            x_pixel = np.sum(x_pixel, axis=1)
            output_array[x, y] = color_array[np.argmin(x_pixel)]

    return output_array

def color_swap(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False):
    # Load image
    img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array)
    
    # output array 
    output_array = np.zeros_like(img_array)
    
    # Run color substitution algorithm            
    output_array = swap_algo(img_array, color_array, output_array)
    
    # rendering
    if render:
        cv2.imshow('Color Palette Swap', np.flip(output_array/256, axis=-1))
        cv2.waitKey(0)


    output_array = np.flip(output_array, axis=-1) 
    opencvimg = cv2.resize(output_array, final_dims, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(main_paths['Outputs'], output_name), opencvimg)
    return output_array

# --------------------------------------------------------------------------- #
# 3.2 Image to color scheme based on k-means clustering
def img_2_color_scheme(input_name, output_name, transparent=True, target_colors=16, render=False, downscale=4, rescale=1, other_array=None, colors=None):
    
    img_array, color_array, final_dims = load_image_and_colors(input_name, colors, downscale=downscale, rescale=rescale, other_array=other_array)
    
    img_array = img_array.reshape((img_array.shape[1]*img_array.shape[0],3))
    kmeans = KMeans(n_clusters=target_colors).fit(img_array)
    labels = list(kmeans.labels_)
    centroid = kmeans.cluster_centers_
    percent=[]
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)
    
    display_img = mpimg.imread(os.path.join(main_paths['Inputs'], input_name))
    
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [4, 1]})
    ax1.axis('off')
    ax2.axis('off')
    ax1.bar(np.arange(len(centroid)),1, 1, color=np.array(centroid/255), label=np.arange(len(centroid)))
    ax2.imshow(display_img)
    fig.savefig(os.path.join(main_paths['Outputs'], output_name), transparent=transparent)
    
    colors = {}
    for i in range(0, len(centroid)):
        colors[i] = centroid[i]
    
    # rendering
    if render:
        cv2.imshow('Color Palette Swap', cv2.imread(os.path.join(main_paths['Outputs'], output_name)))
        cv2.waitKey(0)
    
    return colors


start = time.time()
# 
output_arraya = image_sharpen('mech_sprites.png', 'final_mech_test1.png', color_schemes[19]['colors'], render=False, gamma=1.2)
output_arrayb = bayer_dither('mech_sprites.png', 'final_mech_test0.png', color_schemes[11]['colors'], downscale=2, render=False, kernel_size=2, other_array=output_arraya)
output_array = edge_sharpen('mech_sprites.png', 'final_mech_test2.png', color_schemes[11]['colors'], downscale=1, render=False, algorithm=1, gamma=1, other_array=output_arraya)
end = time.time()
print(end - start)

# img_2_color_scheme('rainy_day.png', 'test3.png')


# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:  
#             pygame.quit()
#             sys.exit()
#             break
