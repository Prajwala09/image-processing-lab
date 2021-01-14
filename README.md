**1.Develop a program to display grayscale image using read and write operation**

In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

To convert an image to grayscale in any of the Microsoft Office suite apps, right-click it and select Format Picture from the context menu . This will open an image editing panel on the right. Go to the Picture tab (the very last one). Expand the Picture Color options, and click the little dropdown next to the Presets for Color Saturation.

import numpy as np
import cv2
img = cv2.imread('nature.jpg',0)
cv2.imshow('Original',img,)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('graynature.jpg',img)

output:
![i1](https://user-images.githubusercontent.com/72255259/104475122-be631100-55e4-11eb-8cb2-0b9717a7312a.jpg)
**2.Develop a program to perform linear transformation on image.**

Linear Transformation is type of gray level transformation that is used for image enhancement. It is a spatial domain method. It is used for manipulation of an image so that the result is more suitable than the original for a specific application.

Image scaling is a computer graphics process that increases or decreases the size of a digital image. An image can be scaled explicitly with an image viewer or editing software, or it can be done automatically by a program to fit an image into a differently sized area.

Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. The input to an image rotation routine is an image, the rotation angle θ, and a point about which rotation is done.

**Scaling:**
import cv2
import numpy as np
src=cv2.imread('original.jpg',1)
img=cv2.imshow('original.jpg',src)
scale_p=500
width=int(src.shape[1]*scale_p/100)
height=int(src.shape[0]*scale_p/100)
dsize=(width,height)
result=cv2.resize(src,dsize)
cv2.imwrite('scaling.jpg',result)
cv2.waitKey(0)
output:

![i2](https://user-images.githubusercontent.com/72255259/104479337-519e4580-55e9-11eb-9283-b74a19c63f5d.jpg)


**rotating:**
import cv2
import numpy as np
src=cv2.imread('original.jpg')
img=cv2.imshow('original.jpg',src)
windowsname='image'
image=cv2.rotate(src,cv2.ROTATE_90_CLOCKWISE)
cv2.imshow(windowsname,image)
c.waitKey(0)
output
![i3](https://user-images.githubusercontent.com/72255259/104480259-64fde080-55ea-11eb-8c24-7d5d0ffd1910.jpg)
**4.Develop a program to convert image to binary image and gray scale.**
Binary images are images whose pixels have only two possible intensity values. Numerically, the two values are often 0 for black, and either 1 or 255 for white. The main reason binary images are particularly useful in the field of Image Processing is because they allow easy separation of an object from the background.

In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

import cv2
img = cv2.imread('original.jpg') 
cv2.imshow('Input',img)
cv2.waitKey(0)
grayimg=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
cv2.imshow('Grayscaleimage',grayimg)
cv2.waitKey(0)
ret, bw_img = cv2.threshold(img,127,255, cv2.THRESH_BINARY) 
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
**output**
![i4](https://user-images.githubusercontent.com/72255259/104538669-5511eb00-5642-11eb-9f93-106abaf7717f.jpg)

**5.Develop a program to convert given color image to different color space.**
Color spaces are different types of color modes, used in image processing and signals and system for various purposes. The color spaces in image processing aim to facilitate the specifications of colors in some standard way. Different types of color spaces are used in multiple fields like in hardware, in multiple applications of creating animation, etc.
import cv2 
image=cv2.imread('original.jpg')
cv2.imshow('pic',image)
cv2.waitKey(0)
yuv_img = cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
cv2.imshow('ychannel',yuv_img[:,:,0])
cv2.imshow('uchannel',yuv_img[:,:,1]) 
cv2.imshow('vchannel',yuv_img[:,:,2])
cv2.waitKey(0) 
hsv_img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
cv2.imshow('hchannel',hsv_img[:,:,0])
cv2.imshow('schannel',hsv_img[:,:,1])
cv2.imshow('vchannel',hsv_img[:,:,2])
cv2.waitKey(0) 
cv2.destroyAllWindows()
**output:**
![Untitled](https://user-images.githubusercontent.com/72255259/104540250-19c4eb80-5645-11eb-9eb9-21deec26b86d.jpg)
![i6](https://user-images.githubusercontent.com/72255259/104540428-76c0a180-5645-11eb-8581-d6b5e7f83a1c.jpg)
![i7](https://user-images.githubusercontent.com/72255259/104540674-ef276280-5645-11eb-8f3c-94a8d6f7193f.jpg)
**6.DEVELOP A PROGRAM TO CREATE AN ARRAY FROM 2D ARRAY**
For a two-dimensional array, in order to reference every element, we must use two nested loops. This gives us a counter variable for every column and every row in the matrix. int cols = 10; int rows = 10; int [] [] myArray = new int [cols] [rows]; // Two nested loops allow us to visit every spot in a 2D array Creating Arrays. You can create an array by using the new operator with the following syntax − Syntax arrayRefVar = new dataType[arraySize]; The above statement does two things − It creates an array using new dataType[arraySize]. It assigns the reference of the newly created array to the variable arrayRefVar.
output:
