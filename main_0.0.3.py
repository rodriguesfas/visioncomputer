# coding: utf-8
# !/usr/bin/python
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
    STACKOVERFLOWS. Improving my Ellipse Fitting Algorithm. Disponível em: <https://stackoverflow.com/questions/21175078/improving-my-ellipse-fitting-algorithm>. Acesso 14/11/2017.
    PYTHONGEEK. Canny Edge Detection. Disponivel em: <http://pythongeek.blogspot.com.br/2012/06/canny-edge-detection.html>. Acesso 07/11/2017.
    EDUCAÇÃO, Mundo. Elipse. Disponível em: <http://mundoeducacao.bol.uol.com.br/matematica/elipse.htm>. Acesso 17/11/2017.
    
@ DataBase:
    ARCHIVE, Computational Vision. Faces 1999 (Front). Disponível em: <http://www.vision.caltech.edu/html-files/archive.html>. Acesso 15/11/2017.
    SPACEK, Libor. Description of the Collection of Facial Images. Disponível em: <http://cswww.essex.ac.uk/mv/allfaces/index.html>. Acesso 13/11/2017.
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


'''
Tempo de execução
'''
START_TIME = time.time()


'''
variáveis globais
'''
SIGMA = 2,2
VIEWIMG = 0


'''
    Configuração do caminho do diretório de imagens.
'''
dir_files = '../src/img/input/database/test/simples'
#dir_files = '../src/img/input/database/test/complexo'

#dir_files = '../src/img/input/database/bd_01/complexo'
#dir_files = '../src/img/input/database/bd_02/complexo'
#dir_files = '../src/img/input/database/bd_02/simples'

img_file = [i for i in glob.glob(dir_files+"/*.jpg")]
img_file.sort()
images = []



'''
    Gerador de código de identificação, para salvar as imagens processadas.
'''
def randomIDImg(size):
    data = '0123456789'
    code = ''
    
    for char in xrange(size):
        code += choice(data)
    
    return  code



'''
    Exibe imagens processadas na tela.
    Usar, somente em modo de teste, quando estiver exibindo as imagens na janela.
    Recomenda-se testar com no MAX. 5 imagens, risco de deadlock.
'''
def showImg(title, img):
    if(VIEWIMG):
        cv2.imshow(title, img)



'''
    Loop
'''
print('Start processing..')

for i in img_file:

    id_img = randomIDImg(10)

    n = cv2.imread(i)
    images.append(n)
    print('file processing: '+i)


    img_orig = cv2.imread(i, 0)
    showImg('Image Original', img_orig)

    # Verifica se a imgagem é RGB antes de converter para escala de cinza.
    if img_orig.shape[-1] == 3:                    # color image
        b, g, r = cv2.split(img_orig)              # get b, g, r
        img_orig = cv2.merge([r, g, b])            # switch it to rgb
        img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

        showImg('Image Cinza', img_gray)
    else:
        img_gray = img_orig



    '''
    Step 01 - Image Enhancement

    As imagens de entrada, podem ter um contraste muito fraco devido à limitação das condições
    de iluminação. Nesse passo, a equalização do histograma é usada para melhorar o contraste 
    da imagem original.
    '''
    img_hist = cv2.equalizeHist(img_gray)
    cv2.imwrite('../src/img/output/processing/'+id_img+'_img_hist.jpg', img_hist)
    showImg('Step 01 - Image Enhancement (Equalize Histogram)', img_hist)



    '''
    Step 02 - Median Filtering (LPF)
     
    Devido ao passo anterior, é obvio que o número de pontos de contorno do rosto podem almentar,
    o que significa, que a informação facial foi fortalecida. O ruído também foi aprimorado. 
    Por meio da filtragem, podem ser apagados as fontes de ruídos presente na imagem, aplicando um
    median filtering.
    '''
    img_hist = cv2.imread('../src/img/output/processing/'+id_img+'_img_hist.jpg', 0)
    img_blur = cv2.medianBlur(img_hist, 5)

    cv2.imwrite('../src/img/output/processing/'+id_img+'_img_blur.jpg', img_blur)
    showImg('Step 02 - Median Filtering', img_blur)



    '''
    Step 03 - Edge Detection

    Existe muitos detectores de bordas, considerando não considerando o custo computacional e o desempenho foi usado
    o zero-crossing detector: Canny Edge Detection (DoG).
    '''

    img_blur = '../src/img/output/processing/'+id_img+'_img_blur.jpg'
    img_blur = Image.open(img_blur)
    img_data = np.array(img_blur, dtype = float)

    img_median = ndi.filters.median_filter(img_data, SIGMA)

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

    scipy.misc.imsave('../src/img/output/processing/'+id_img+'_img_cannynewmag.jpg', sobeloutmag)
    scipy.misc.imsave('../src/img/output/processing/'+id_img+'_img_cannynewdir.jpg', sobeloutdir)

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

    scipy.misc.imsave('../src/img/output/processing/'+id_img+'_img_cannynewdirquantize.jpg', sobeloutdir)



    '''
    Step 04 - Edge Linking

    Conecta as bordas pequenas da imagem com as bordas grandes, com base na direção das bordas, por meio do gradiene de bordas com 
    o operador Sobel.
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

    scipy.misc.imsave('../src/img/output/processing/'+id_img+'_img_cannynewmagsup.jpg', mag_sup)

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

    scipy.misc.imsave('../src/img/output/processing/'+id_img+'.jpg', gnl)

    gnl = gnl-gnh

    scipy.misc.imsave('../src/img/output/processing/'+id_img+'_img_cannynewgnlafterminus.jpg', gnl)
    scipy.misc.imsave('../src/img/output/processing/'+id_img+'_img_cannynewgnh.jpg', gnh)



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

    scipy.misc.imsave('../src/img/output/processing/'+id_img+'_img_cannynewout.jpg', gnh)

    showImg('Step 03/04 - Edge Detection/Edge Linking', gnh)



    '''
    Step 05 - Template Matching

    '''
    cont_face = 0
    edge_image = cv2.imread('../src/img/output/processing/'+id_img+'_img_cannynewout.jpg', 0)
    
    img_dilate = cv2.dilate(edge_image, np.ones((3, 3)), iterations=1)
    #img_eroded = cv2.erode(img_dilate, np.ones((3, 3)), iterations=7)

    showImg("Dilate", img_dilate)

    labels, count = ndi.label(img_dilate)

    for lab, idx in enumerate(ndi.find_objects(labels.astype(int)), 1):
        sy = idx[0].start
        sx = idx[1].start
        y, x = np.where(labels[idx] == lab)
        ellp = cv2.fitEllipse(np.column_stack((x+sx, y+sy)))

        '''
            @ Parâmetros de uma elipse
                
                xc: coordenada x do centro
                yc: coordenada y do centro
                a: semi-eixo principal
                b: semi-eixo secundário
                theta: ângulo de rotação
        '''
        (xc, yc), (a, b), theta = cv2.fitEllipse(np.column_stack((x+sx, y+sy)))
        
        # Verificar se é uma elipse.
        isElipse = ( (xc**2) / (b**2) ) + ( (yc**2) / (a**2) )

        # Mostra caracgteristicas da elipse.
        print("value: ", isElipse)
        print("xc: ", xc)
        print("yc: ", yc) 
        print("a: ", a)
        print("b: ", b)
        print("theta: ", theta)
        print("ellp", ellp)

        if(isElipse <= 1):
            img_orig = cv2.ellipse(img_orig, ellp, (255, 255, 255), 1)
            img_dilate = cv2.ellipse(img_dilate, ellp, (255, 255, 255), 1)
            cont_face+=1
            print('É uma elipse :) ')
            print "I found {0} face(s) in that image".format(cont_face)
        else: 
            print('Não é uma elipse :( ')

    scipy.misc.imsave('../src/img/output/result/'+id_img+'_img_detection.jpg', img_orig)
    scipy.misc.imsave('../src/img/output/result/'+id_img+'_img_detection2.jpg', img_dilate)

    showImg("Step 05 - Template Matching", img_orig)



END_TIME = time.time()
TIME_TAKEN = END_TIME - START_TIME

print('finish')
print 'Time taken for execution: ', TIME_TAKEN



'''
Exit
Usar, somente em modo de teste, quando estiver exibindo as imagens na janela.
'''
#cv2.waitKey(0)
#cv2.destroyAllWindows()
