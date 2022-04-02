# Stain_visualization

You could use this tool to visualize the stain (i.e., color) pattern of hunderds of images.

You will see an example in example.py

First, you need to gather existing images via get_file_list method.
Then you can use image_color_manifold to analyze the stain pattern of these images.

If you have the xlsxwriter python library, you will got an Excel sheet. Within which the color is showed intuitively:

![](https://github.com/jiaoyiping630/Stain_visualization/Public/blob/master/color_spectrum_in_Excel.png) 

Moreover, you will find something in the working directory you assigned, including color bars, and a mat file.
Then you should use MATLAB (code already attached in example.py), and pass the mat file as the input (namely manifold_path in MATLAB function visualization).
Then you will see the spatial distribution of these color bars:

![](https://github.com/jiaoyiping630/Stain_visualization/Public/blob/master/color_spectrum_in_Matlab.jpg) 
