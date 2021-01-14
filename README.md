**1.Develop a program to display grayscale image using read and write operation**

In digital photography, computer-generated imagery, and colorimetry, a grayscale or image is one in which the value of each pixel is a single sample representing only an amount of light; that is, it carries only intensity information. Grayscale images, a kind of black-and-white or gray monochrome, are composed exclusively of shades of gray.

To convert an image to grayscale in any of the Microsoft Office suite apps, right-click it and select Format Picture from the context menu . This will open an image editing panel on the right. Go to the Picture tab (the very last one). Expand the Picture Color options, and click the little dropdown next to the Presets for Color Saturation.

import cv2
image=cv2.imread('original.jpg')
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('original.jpg',image) 
cv2.imshow("org",image)
cv2.imshow("gimg",grey_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows()
output:
![3](https://user-images.githubusercontent.com/72255259/104600668-4d346400-569f-11eb-8e45-6bc2e22d18bd.jpg)


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
**3.Create a program to find sum and mean of a set of image.**
In digital image processing, the sum of absolute differences (SAD) is a measure of the similarity between image blocks. It is calculated by taking the absolute difference between each pixel in the original block and the corresponding pixel in the block being used for comparison

Mean is most basic of all statistical measure. Means are often used in geometry and analysis; a wide range of means have been developed for these purposes. In contest of image processing filtering using mean is classified as spatial filtering and used for noise reduction.
import cv2
import os 
path='C:\picture\images' 
imgs=[]
dirs=os.listdir(path)
for file in dirs: 
    fpat=path+"\\"+file
    imgs.append(cv2.imread(fpat))
i=0 
sum_img=[] 
for sum_img in imgs: 
    read_imgs=imgs[i] 
    sum_img=sum_img+read_imgs #cv2.imshow(dirs[i],imgs[i]) 
    i=i+1 
    print(i) 
    cv2.imshow('sum',sum_img)
    print(sum_img)
cv2.imshow('mean',sum_img/i)
mean=(sum_img/i)
print(mean)
cv2.waitKey()
cv2.destroyAllwindows()
**output**
![1](https://user-images.githubusercontent.com/72255259/104597871-e2355e00-569b-11eb-93e1-00816649ad86.jpg)

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

import numpy as np 
from PIL import Image
import cv2
array = np.linspace(0,1,256*256)
mat = np.reshape(array,(256,256))
img = Image.fromarray(np.uint8(mat * 255) , 'L')
img.show() 
cv2.waitKey(0) 
array = np.linspace(0,1,256*256)
mat = np.reshape(array,(256,256))
img = Image.fromarray( mat , 'L')
img.show() 
cv2.waitKey(0)
**output:**
![c2](https://user-images.githubusercontent.com/72255259/104577054-09316700-567f-11eb-95bf-8321556a7105.jpg)

**7.Find the neighbour of matrices**

In topology and related areas of mathematics, a neighbourhood (or neighborhood) is one of the basic concepts in a topological space.It is closely related to the concepts of open set and interior.Intuitively speaking, a neighbourhood of a point is a set of points containing that point where one can move some amount in any direction away from that point without leaving the set.

import numpy as np
i=0
j=0
a= np.array([[1,2,3,4,5], [2,3,4,5,6],[3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9]])
print("a : ",str(a))
def neighbors(radius, rowNumber, columnNumber):
     return [[a[i][j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
                for j in range(columnNumber-1-radius, columnNumber+radius)]
                    for i in range(rowNumber-1-radius, rowNumber+radius)]
neighbors(1, 2, 3)

**output:**
a :  [[1 2 3 4 5]
 [2 3 4 5 6]
 [3 4 5 6 7]
 [4 5 6 7 8]
 [5 6 7 8 9]]
[[2, 3, 4], [3, 4, 5], [4, 5, 6]]


**8.SUM OF NEIGHBORS**

Given a M x N matrix, find sum of all K x K sub-matrix 2. Given a M x N matrix and a cell (i, j), find sum of all elements of the matrix in constant time except the elements present at row i & column j of the matrix. Given a M x N matrix, calculate maximum sum submatrix of size k x k in a given M x N matrix in O (M*N) time. Here, 0 < k < M, N.

import numpy as np
def sumNeighbors(M,x,y):
    l = []
    for i in range(max(0,x-1),x+2): # max(0,x-1), such that no negative values in range() 
        for j in range(max(0,y-1),y+2):
            try:
                t = M[i][j]
                l.append(t)
            except IndexError: # if entry doesn't exist
                pass
    return sum(l)-M[x][y] # exclude the entry itself
M = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]] 
M = np.asarray(M)
N = np.zeros(M.shape)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        N[i][j] = sumNeighbors(M, i, j)
print("Original matrix:\n",M)
print("Summed neighbors matrix:\n",N)

**output**
Original matrix:
 [[1 2 3]
 [4 5 6]
 [7 8 9]]
Summed neighbors matrix:
 [[11. 19. 13.]
 [23. 40. 27.]
 [17. 31. 19.]]
