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


![dithering_comparison](https://github.com/Lagrange417/pixelartgenerator/assets/134843622/a4b30514-719d-4b18-855b-0ff93303680b)


**1.1 Basic Dithering**

This module has the implmentation for nine different dithering matrices. These are `0: Floyd Steinberg, 1: False Floyd Steinberg, 2: Jarvis, Judice, Ninke, 3: Stucki 4: Atkinson, 5: Burkes, 6: sierra, 7: Two-Row Sierra, 8: Sierra Lite`. See reference image for comparison of the results of the different algorithms. 

`dithering(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, algorithm=0)`
- `algorithm`: determines the dithering matrix. Available are `0: Floyd Steinberg, 1: False Floyd Steinberg, 2: Jarvis, Judice, Ninke, 3: Stucki 4: Atkinson, 5: Burkes, 6: sierra, 7: Two-Row Sierra, 8: Sierra Lite`

**1.2 Bayer Dithering** 

`bayer_dither(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, kernel_size=2, render=False)`: This module implements 2x2, 4x4, and 8x8 Bayer Dithering. 
- `kernel_size`: Determines the size of the bayer kernel. The following options are avaiable: `2: 2x2 Bayer Kernel, 4: 4x4 Kernel, 8: 8x8 Kernel`

**1.3 Interleaved Gradient Noise Dithering** 

This module implements Interleaved Gradient Noise (IGN) Dithering with two different initial default values. I recommend playing around with https://observablehq.com/@jobleonard/pseudo-blue-noise to find more default settings for IGN dithering. 

`ign_dither(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, ign_preset=0)`
- `ign_presets`: Dict_like, the formula for IGN dithering is `ign_preset[0]` * (ign_preset[1] * x + ign_preset[2] * y). Different presets result in different patters. Refer to the link mentioned earlier to get a feel for the resulting pattern. 

**1.4. Blue Noise Dithering**

This modules uses a blue noise texture to create its dithering pattern. There are three different blue noise textures in the `pixelartgenerator/resources` folder. The sizes are 16x16, 32x32, and 64x64. I recommend using the largest format (64x64) to avoid generating visible repeating patterns. As a small bonus, I added a 128x128 texture. you can replace that file with any random image file to blend a texture with an image.  

`blue_noise_dither(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, kernel_size=64)`
- `kernel_size`: Determines the size of the blue_noise texture. 16x16, 32x32, and 64x64 implemented


### 2.0 Edge Sharpening and Overlay 

**Note:** This feature a bit experimental and finicky. Basically it detects edges using the Sobel, Prewitt, or Sharr operator. It then overlays the edge on the image using the color palette's darkest and lightest number. Results may vary depending on the input image...

`edge_sharpen(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, algorithm=0, gamma=1)`
- algorithm refers to the Sobel, Prewitt, or Sharr operator. `0=Sobel, 1=Prewitt, 2=Sharr`

### 3.0 Image Sharpening 

This effect uses two different types of sharpening kernels to sharpen the image. 'Gamma' is a paramenter that one can use to brighten/darken the image. Again its a bit experimental and results may vary. 

`image_sharpen(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False, gamma=1, algorithm=0)`

the kernels used are:

$$ 0:
\left(\begin{array}{cc} 
0 & -1 & 0 \\
-1 & 5 & -1 \\
0 & -1 & 0
\end{array}\right),  1: 
\left(\begin{array}{cc} 
-1 & -1 & -1 \\
-1 & 9 & -1 \\
-1 & -1 & -1
\end{array}\right)
$$

### 4.0 Other Tools

**4.1 Color Palette Swap**
Substitues the colors in an image with the closest color from another palette. 

`color_swap(input_name, output_name, colors, downscale=1, rescale=1, other_array=None, render=False)`

**4.2 Image to Color Palette**
Uses KMeans clustering to find the main colors that make up the image. Returns a color palette in dict-fomat of `{1: (R, G, B), 2: (R, G, B) ...}`. Also it outputs an image with the color palette below because I thought that would be cute :) 

**NOTE:** Unlike the other modules this is not very optimized. Recommend downscaling the image to ~200x300 using the downscale paramenter to make it faster. The output vignette will not be affected by the downscale. 

`img_2_color_scheme(input_name, output_name, transparent=True, target_colors=16, render=False, downscale=4, rescale=1, other_array=None, colors=None)`
- `target_colors`: Number of colors that will be part of the output color palette in the format of `{1: (R, G, B), 2: (R, G, B) ...}`

![palette_examples](https://github.com/Lagrange417/pixelartgenerator/assets/134843622/4c990767-9eac-4b0a-9ba1-81ae953d82de)


### Apendix

**More Dithering Comparisons**

![dithering_comparison2](https://github.com/Lagrange417/pixelartgenerator/assets/134843622/b7927bb5-4c61-4076-b982-e70b04fa3c41)


