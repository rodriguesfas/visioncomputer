#coding: utf-8
#!/usr/bin/python
# Python 2/3 compatibility

'''

@ Article Title: A new face detection method based on shape information
@ Article Url: <http://www.sciencedirect.com/science/article/pii/S0167865500000088>
@ Article Implemented by Francisco de Assis de Souza Rodrigues - RodriguesFAS
@ Date 21/10/2017
@ Email <fasr@cin.ufpe.br> || <franciscosouzaacer@gmail.com>
@ Website <htpp://rodriguesfas.com.br>

@ References
    DOC, OpenCV. Histogram Equalization. Disponível em: <https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html>. Acesso 21/10/2017.
    DOC, OpenCV. Smoothing Images. Disponível em: <https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html>. Acesso 21/10/2017.
    HIPR2. Zero Crossing Detector. Disponível em: <https://homepages.inf.ed.ac.uk/rbf/HIPR2/zeros.htm>. Acesso 30/10/2017.
    HONG. IMAGE EDGE DETECTION : SOBEL AND LAPLACIAN. Disponível em: <http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Gradient_Sobel_Laplacian_Derivatives_Edge_Detection.php>. Acesso 2181082017.

'''



# Import lib's.
import cv2
import numpy as np
import glob
import scipy.ndimage as ndi
import scipy
import Image
import math
from math import pi
import time
from random import choice



start_time = time.time()



sigma = 2
VIEWIMG = 0


# read img_orig
filenames = [i for i in glob.glob("../src/img/input/*.jpg")]
filenames.sort() # ADD THIS LINE

images = []

'''
Gerador de ID para salvar as imagens processadas.
'''
def randomIDImg(size):
    data = '0123456789'
    code = ''
    
    for char in xrange(size):
        code += choice(data)
    
    return  code


'''
Exibe imagens processadas na tela.
'''
def showImg(title, img):
    if(VIEWIMG):
        cv2.imshow(title, img)



'''
Loop
'''
print('processing..')

for i in filenames:

    id_img = randomIDImg(10)

    n = cv2.imread(i)
    images.append(n)
    print('file: '+i)


    img_orig = cv2.imread(i, 0)
    showImg('Image Original', img_orig)



    '''
    Step 01 - Image Enhancement

    As imagens de entrada, podem ter um contraste muito fraco devido à limitação das condições
    de iluminação. Nesse passo, a equalização do histograma é usada para melhorar o contraste 
    da imagem original.
    '''
    img_hist = cv2.equalizeHist(img_orig)
    cv2.imwrite('../src/img/output/'+id_img+'_img_hist.jpg', img_hist)
    showImg('Step 01 - Image Enhancement (Equalize Histogram)', img_hist)



    '''
    Step 02 - Median Filtering (LPF)
     
    Devido ao passo anterior, é obvio que o número de pontos de contorno do rosto podem almentar,
    o que significa, que a informação facial foi fortalecida. O ruído também foi aprimorado. 
    Por meio da filtragem, podem ser apagados as fontes de ruídos presente na imagem, aplicando um
    median filtering.
    '''
    img_hist = cv2.imread('../src/img/output/'+id_img+'_img_hist.jpg', 0)
    img_blur = cv2.medianBlur(img_hist, 5)

    cv2.imwrite('../src/img/output/'+id_img+'_img_blur.jpg', img_blur)
    showImg('Step 02 - Median Filtering', img_blur)



    '''
    Step 03 - Edge Detection

    Existe muitos detectores de bordas, considerando não considerando o custo computacional e o desempenho foi usado
    o zero-crossing detector: Canny Edge Detection (DoG).
    '''

    img_blur = '../src/img/output/'+id_img+'_img_blur.jpg'
    img_blur = Image.open(img_blur)
    img_data = np.array(img_blur, dtype = float)

    img_median = ndi.filters.median_filter(img_data, sigma)

    # imagem vazia
    sobelout = Image.new('L', img_blur.size)
    gradx = np.array(sobelout, dtype = float)                        
    grady = np.array(sobelout, dtype = float)

    sobel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
        ]

    sobel_y = [
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
        ]

    width = img_blur.size[1]
    height = img_blur.size[0]

    #calculate |img_median| and dir(img_median)

    for x in range(1, width-1):
        for y in range(1, height-1):
            px = (sobel_x[0][0] * img_median[x-1][y-1]) + (sobel_x[0][1] * img_median[x][y-1]) + \
                 (sobel_x[0][2] * img_median[x+1][y-1]) + (sobel_x[1][0] * img_median[x-1][y]) + \
                 (sobel_x[1][1] * img_median[x][y]) + (sobel_x[1][2] * img_median[x+1][y]) + \
                 (sobel_x[2][0] * img_median[x-1][y+1]) + (sobel_x[2][1] * img_median[x][y+1]) + \
                 (sobel_x[2][2] * img_median[x+1][y+1])

            py = (sobel_y[0][0] * img_median[x-1][y-1]) + (sobel_y[0][1] * img_median[x][y-1]) + \
                 (sobel_y[0][2] * img_median[x+1][y-1]) + (sobel_y[1][0] * img_median[x-1][y]) + \
                 (sobel_y[1][1] * img_median[x][y]) + (sobel_y[1][2] * img_median[x+1][y]) + \
                 (sobel_y[2][0] * img_median[x-1][y+1]) + (sobel_y[2][1] * img_median[x][y+1]) + \
                 (sobel_y[2][2] * img_median[x+1][y+1])
            gradx[x][y] = px
            grady[x][y] = py

    sobeloutmag = scipy.hypot(gradx, grady)
    sobeloutdir = scipy.arctan2(grady, gradx)

    scipy.misc.imsave('../src/img/output/'+id_img+'_img_cannynewmag.jpg', sobeloutmag)
    scipy.misc.imsave('../src/img/output/'+id_img+'_img_cannynewdir.jpg', sobeloutdir)

    for x in range(width):
        for y in range(height):
            if (sobeloutdir[x][y]<22.5 and sobeloutdir[x][y]>=0) or \
               (sobeloutdir[x][y]>=157.5 and sobeloutdir[x][y]<202.5) or \
               (sobeloutdir[x][y]>=337.5 and sobeloutdir[x][y]<=360):
                sobeloutdir[x][y]=0
            elif (sobeloutdir[x][y]>=22.5 and sobeloutdir[x][y]<67.5) or \
                 (sobeloutdir[x][y]>=202.5 and sobeloutdir[x][y]<247.5):
                sobeloutdir[x][y]=45
            elif (sobeloutdir[x][y]>=67.5 and sobeloutdir[x][y]<112.5)or \
                 (sobeloutdir[x][y]>=247.5 and sobeloutdir[x][y]<292.5):
                sobeloutdir[x][y]=90
            else:
                sobeloutdir[x][y]=135

    scipy.misc.imsave('../src/img/output/'+id_img+'_img_cannynewdirquantize.jpg', sobeloutdir)



    '''
    Step 04 - Edge Linking

    Conecta as bordas pequenas da imagem com as bordas grandes, com base na direção das bordas.
    '''
    mag_sup = sobeloutmag.copy()

    for x in range(1, width-1):
        for y in range(1, height-1):
            if sobeloutdir[x][y]==0:
                if (sobeloutmag[x][y]<=sobeloutmag[x][y+1]) or \
                   (sobeloutmag[x][y]<=sobeloutmag[x][y-1]):
                    mag_sup[x][y]=0
            elif sobeloutdir[x][y]==45:
                if (sobeloutmag[x][y]<=sobeloutmag[x-1][y+1]) or \
                   (sobeloutmag[x][y]<=sobeloutmag[x+1][y-1]):
                    mag_sup[x][y]=0
            elif sobeloutdir[x][y]==90:
                if (sobeloutmag[x][y]<=sobeloutmag[x+1][y]) or \
                   (sobeloutmag[x][y]<=sobeloutmag[x-1][y]):
                    mag_sup[x][y]=0
            else:
                if (sobeloutmag[x][y]<=sobeloutmag[x+1][y+1]) or \
                   (sobeloutmag[x][y]<=sobeloutmag[x-1][y-1]):
                    mag_sup[x][y]=0

    scipy.misc.imsave('../src/img/output/'+id_img+'_img_cannynewmagsup.jpg', mag_sup)

    m = np.max(mag_sup)
    th = 0.2*m
    tl = 0.1*m

    gnh = np.zeros((width, height))
    gnl = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            if mag_sup[x][y]>=th:
                gnh[x][y]=mag_sup[x][y]
            if mag_sup[x][y]>=tl:
                gnl[x][y]=mag_sup[x][y]

    scipy.misc.imsave('../src/img/output/'+id_img+'.jpg', gnl)

    gnl = gnl-gnh

    scipy.misc.imsave('../src/img/output/'+id_img+'_img_cannynewgnlafterminus.jpg', gnl)
    scipy.misc.imsave('../src/img/output/'+id_img+'_img_cannynewgnh.jpg', gnh)



    def traverse(i, j):
        x = [-1, 0, 1, -1, 1, -1, 0, 1]
        y = [-1, -1, -1, 0, 0, 1, 1, 1]
        for k in range(8):
            if gnh[i+x[k]][j+y[k]]==0 and gnl[i+x[k]][j+y[k]]!=0:
                gnh[i+x[k]][j+y[k]]=1
                traverse(i+x[k], j+y[k])

    for i in range(1, width-1):
        for j in range(1, height-1):
            if gnh[i][j]:
                gnh[i][j]=1
                traverse(i, j)

    scipy.misc.imsave('../src/img/output/'+id_img+'_img_cannynewout.jpg', gnh)

    showImg('Step 03/04 - Edge Detection/Edge Linking', gnh)



    '''
    Step 05 - Template Matching

    '''
    edge_image = cv2.imread('../src/img/output/'+id_img+'_img_cannynewout.jpg', 0)
    
    img_dilate = cv2.dilate(edge_image, np.ones((3, 3)), iterations=1)
    cv2.imshow('Dilate', img_dilate)

    labels, count = ndi.label(img_dilate)

    for lab, idx in enumerate(ndi.find_objects(labels.astype(int)), 1):
        sy = idx[0].start
        sx = idx[1].start
        y, x = np.where(labels[idx] == lab)
        ellp = cv2.fitEllipse(np.column_stack((x+sx, y+sy)))
        cv2.ellipse(img_orig, ellp, (0, 255, 0))

    scipy.misc.imsave('../src/img/output/result/'+id_img+'_img_detection.jpg', img_orig)
    cv2.imshow('Step 05 - Template Matching', img_orig)



end_time = time.time()
time_taken = end_time - start_time
print "Time taken for execution: ", time_taken



# Exit
#cv2.waitKey(0)
#cv2.destroyAllWindows()
