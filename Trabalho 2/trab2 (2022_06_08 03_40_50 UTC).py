''' João Victor Dell Agli Floriano - 10799783
    SCC0251 Processamento de Imagens, 2022/1'''
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio

def bit8_convert(img):
  '''Function that converts an image to 8-bit range.'''
  return ((img - np.min(img))/(np.max(img) - np.min(img)))*(2**8 - 1.0)



#Limiarization functions
def limiarization(img, T):

  '''Given a threshold T0, separates the image into two groups, 
     one lower than T0 and other higher. Each reagion is capped to 1 or 0.'''

  M, N = img.shape

  img1 = np.zeros((M, N))

  for i in range(M):
    for j in range(N):
      if img[i][j] > T:
        img1[i][j] = 1
      else:
        img1[i][j] = 0

  return img1


def optimal_T(img, T_0):

  '''Finds the optimal threshold T using an initial T_0.'''

  T = 0
  Tp = 1

  M, N = img.shape

  img = limiarization(img, T_0)

  while( np.abs(T - Tp) >= 0.5):

    avg1 = 0
    avg2 = 0
    count1 = 0
    count2 = 0

    for i in range(M):
      for j in range(N):
        if( img[i][j] == 0 ):
          avg1 += img[i][j]
          count1 += 1
        else: 
          avg2 += img[i][j]
          count2 += 1

    avg1 = avg1/count1
    avg2 = avg2/count2

    Tp = T

    T = 0.5*(avg1 + avg2)

    img = limiarization(img, T)

  return T


def filter1D(img, n, weights):

    '''Applies a one-dimensional filter with a weights array of size n.'''

    M, N = img.shape

    img1 = np.zeros(M*N)

    imgflat = img.flatten( order = 'C')

    offset = n - (1 + n // 2)
    aux = np.pad(imgflat, (offset, offset), mode = "wrap")


    for i in range(M*N):
        for j in range(weights.shape[0]):
            img1[i] += weights[j]*aux[i + j] 
        
    
    img1 = np.reshape(img1, (M, N))

    return img1


def filter2D(img, n, weights_mat, T0):

    '''Applies a two-dimensional filter with a weights matrix of size n*n.'''

    M, N = img.shape

    img1 = np.zeros((M,N))

    offset = n - (1 + n // 2)
    aux = np.pad(img, (offset, offset), 'edge')


    for i in range(0, M):
        for j in range(0, N):
            img1[i][j] += np.trace( np.matmul(weights_mat, np.transpose(aux[i : (i + n), j : (j + n)])) )


    T = optimal_T(img1, T0)
    img1 = limiarization(img1, T)

    return img1


def median_filter(img, n):

    '''Applies a filter that replaces an image pixel with the median intensity level of its neighbourhood.'''

    M, N = img.shape

    offset = n - (1 + n // 2) 

    aux = np.pad(img, (offset, offset), "constant", constant_values = 0)

    img1 = np.zeros((M, N))

    for i in range(offset, M + offset):
        for j in range(offset, N + offset):
            aux1 = aux[ i - offset : i + offset, j - offset : j + offset ]
            img1[i - offset][j - offset] = np.median( aux1.flatten( order  = 'C') )


    return img1



#1 - PARAMETER INPUT

T_0 = 0
n = 0
weights = np.zeros(n)
weights_matrix = np.zeros((n,n))


#image load
im_path = str(input()).rstrip()

img = iio.imread(im_path)

M, N = img.shape



#method collection
method = int(input())
assert method in (1, 2, 3, 4), "Method to be used must be an integer bewteen 1 and 4!\n"





if(method == 1):
  T_0 = int(input())
  assert T_0 in range(0, 256), "Initial threshold must be an integer between 0 and 255!\n"

  T = optimal_T(img, T_0)

  img1 = limiarization(img, T)




elif(method == 2):
  n = int(input())
  assert type(n) == int, "Filter size must be an integer!\n"

  weis = input().split(" ")
  assert len(weis) == n, "Number of weights must be " + str(n)
  
  for i in range(n):
    weights = np.append(weights, np.array(weis[i]))

  weights = weights.astype(float) 


  img1 = filter1D(img, n, weights)





elif(method == 3):
  n = int(input())
  assert type(n) == int, "Filter size must be an integer!\n"

  for i in range(n):
    weis = input().split(" ")
    assert len(weis) == n, "Number of weights must be " + str(n)

    for j in range(n):
      weights_matrix = np.append(weights_matrix, np.array(weis[j]))


  weights_matrix = weights_matrix.astype(float) 

  weights_matrix = np.reshape(weights_matrix, (n, n))

  T_0 = int(input())
  assert T_0 in range(0, 256), "Initial threshold must be an integer between 0 and 255!\n"

  img1 = filter2D(img, n, weights_matrix, T_0)


elif(method == 4):
  n = int(input())
  assert type(n) == int, "Filter size must be an integer!\n"


  img1 = median_filter(img, n)




# RMSE CALCULATION

img1 = bit8_convert(img1)
img1 = img1.astype(np.uint8)

RMSE = np.sqrt((np.sum((img - img1)**2))/(float(M*N)))

print('%0.4f' % RMSE)
