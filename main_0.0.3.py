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
from matplotlib import pyplot as plt
from scipy import signal
import time



# outHist - Exibe resultado do histograma da imagem em questão.
def outHist(img, title):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.title(title, loc='left')
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()



# Input image - 
def loadImg():
    img_orig = cv2.imread('../src/img/input/eu4.jpg', 0)
    cv2.imshow('Image Original', img_orig)
    #outHist(img_orig, "Image Original")
    return img_orig



'''
Step 01 - Image Enhancement

As imagens de entrada, podem ter um contraste muito fraco devido à limitação das condições
de iluminação. Nesse passo, a equalização do histograma é usada para melhorar o contraste 
da imagem original.
'''
def step_01(img):
    img_hist = cv2.equalizeHist(img)
    cv2.imwrite('../src/img/output/img_hist.png', img_hist)
    cv2.imshow('Step 01 - Image Enhancement (Equalize Histogram)', img_hist)
    #outHist(img_hist, "Step 01 - Image Enhancement (Equalize Histogram)")
    return img_hist



'''
Step 02 - Median Filtering (LPF)
 
Devido ao passo anterior, é obvio que o número de pontos de contorno do rosto, podem almentar,
o que significa, que a informação facial foi fortalecida. O ruído também foi aprimorado. 
Por meio da filtragem, podem ser apagados as fontes de ruídos presente na imagem, aplicando um
filtro de passa mediana.
'''
def step_02(img):
    # Median Blurring
    img_blur = cv2.medianBlur(img, 5)

    cv2.imwrite('../src/img/output/img_blur.png', img_blur)
    cv2.imshow('Step 02 - Median Filtering', img_blur)

    #outHist(img_blur)

    return img_blur



'''
Step 03/04 - Edge Detection/Edge Linking

Existe muitos detectores de bordas, considerando o custo computacional e o desempenho foi usado
o zero-crossing detector: (DoG).
'''
def step_03_04(img, sigma):

    # calcular a mediana das intensidades de pixel do canal único.
    v = np.median(img)

    # aplica a detecção automática de borda Canny usando a mediana calculada.
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    img_edged = cv2.Canny(img, lower, upper)

    wide = cv2.Canny(img, 10, 200)
    tight = cv2.Canny(img, 225, 250)

    cv2.imwrite('../src/img/output/img_edged.png', img_edged)

    cv2.imshow('Step 03 - Edge Detection Canny (DoG Z-Crossing)', img_edged)
    #cv2.imshow("Step 03 - Edges", np.hstack([wide, tight, img_edged]))

    return img_edged



# Step 05 - Template Matching
def step_05(img):

    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]), i[2],(0, 255, 0), 1)
        
        # draw the center of the circle
        #cv2.circle(cimg,(i[0],i[1]), 2, (0,0,255), 3)

    cv2.imshow('Step 05 - Template Matching', cimg)



# Exit
def exit():
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# Main
def main():
    start_time = time.time()

    img_orig = loadImg()
    img_hist = step_01(img_orig)
    img_blur = step_02(img_hist)
    img_edge_link = step_03_04(img_blur, sigma=0.33)
    step_05(img_edge_link)

    end_time = time.time()
    time_taken = end_time - start_time

    print "Time taken for execution: ", time_taken

    exit()



# Start
if __name__ == "__main__": main()