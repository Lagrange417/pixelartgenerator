# Pixel Art Generator
Python Library with Tools for Converting Images to Pixel Art


### 0.0 General Function Strucutre 

Most modules are set up in the following manner:

- `input_name`: name of the input image in `pixelartgenerator/inputs`
- `output_name`: filename to which the output image will be stored in `pixelartgenerator/outputs`
- `color_array`: dict of the form `{index_0: (R, G, B),  index_1: (R, G, B), ....}`
- `downscale`: will resize the image by dividing the width and height of the image by the downscale factor, i.e. with `downscale=2` the image will be half a large. Resizing helps achieve a more 'pixel art' appearance
- `rescale`: Rescale factor of 1 will resize the pixel size to their original size. The algorithm uses `cv2.INTER_NEAREST` so there is no interpolation or weighting of pixels to achieve a more 'pixel art' appearance
- `other_array`: most modules output a CV2 image array. Passing that array to anoter function will overwrite the import from .png and will allow to sequentially process the image
- `render`: chose to display the image via CV2 or not. Recommend turning render off for sequential processing, since it interrupts the code execution until the window is closed. 

### 0.1 Utility Modules 

**Color Scheme Importer**

`color_scheme_import(palette_size=32)`: Function to import lospec color palettes from the `pixelartgenerator/color_scheme` folder
- `palette_size`: Lospec palettes come in differnt pixel sizes. I use 32x32 color palettes with the default palettes


**Image Loader**

`load_image_and_colors(input_name, colors, downscale=1, rescale=, other_array=None)`: Image loader function. Most modules make use of that function. Usually you should not interact with this function. 



### 1.0 Dithering Algorithms 

**1.1 Basic Dithering**

This module has the implmentation for nine different dithering matrices. These are `0: Floyd Steinberg, 1: False Floyd Steinberg, 2: Jarvis, Judice, Ninke, 3: Stucki 4: Atkinson, 5: Burkes, 6: sierra, 7: Two-Row Sierra, 8: Sierra Lite`. See reference image for comparison of the results of the different algorithms. 

**1.2 Bayer Dithering** 

This module implements 2x2, 4x4, and 8x8 Bayer Dithering. 

**1.3 Interleaved Gradient Noise Dithering** 

This module implements Interleaved Gradient Noise (IGN) Dithering with two different initial default values. I recommend playing around with https://observablehq.com/@jobleonard/pseudo-blue-noise to find more default settings for IGN dithering. 
