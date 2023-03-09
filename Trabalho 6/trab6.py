''' João Victor Dell Agli Floriano - 10799783
    SCC0251 Processamento de Imagens, 2022/1'''
import numpy as np
import imageio as iio

def RMSE(img, img1):
  '''Function that calculates the RMSE between img and img1.'''
  return np.sqrt((np.sum((img - img1)**2))/(float(img.shape[0]*img.shape[1])))

def normalize(img : np.ndarray, min = 0.0, max = 1.0):
  '''Function that converts an image to the given desired range.'''
  return ((img - np.min(img))/(np.max(img) - np.min(img)))*(max - min) + min

def euclidian_distance(array1 : np.ndarray, array2 : np.ndarray, axis : int = 0):
  return np.sqrt( np.sum( np.square( array1 - array2 ), axis = axis) )

def luminance(color : np.ndarray):
  return 0.299*color[0] + 0.587*color[1] + 0.114*color[2]


def kmeans(img, centroids0, k, n, op):
  attrib_matrix = np.zeros((k, img.shape[0], img.shape[1]), dtype = np.float32)

  for j in range(n):
    for i in range(k):

      if op == 1:
        cent_matrix = np.full(img.shape, centroids0[i], dtype = np.float32)
        attrib_matrix[i] = euclidian_distance(img.astype(np.float32), cent_matrix.astype(np.float32), axis = 2)
      elif op == 2:
        cent_matrix = np.full(img.shape, centroids0[i], dtype = np.float32)
        attrib_matrix[i] = euclidian_distance(img.astype(np.float32), cent_matrix.astype(np.float32), axis = 2)
      elif op == 3:
        cent_matrix = np.full(img.shape, centroids0[i], dtype = np.float32)
        attrib_matrix[i] = euclidian_distance(img.astype(np.float32), cent_matrix.astype(np.float32), axis = 2)
      elif op == 4:
        cent_matrix = np.full(img.shape, centroids0[i], dtype = np.float32)
        attrib_matrix[i] = euclidian_distance(img.astype(np.float32), cent_matrix.astype(np.float32), axis = 2)


    att_min = attrib_matrix.min(axis = 0)
    att_min2 = np.zeros(img.shape)


    for i in range(img.shape[0]):
      for h in range(img.shape[1]):
        att_min2[i, h] = np.where( attrib_matrix[:,i, h] == attrib_matrix.min(axis = 0)[i, h])[0][0]


    att_min2 = att_min2.astype(np.uint8)


    mask = np.zeros((img.shape), dtype = np.float32)

    for i in range(k):
      mask[att_min2 == i] = img[att_min2 == i]
      centroids0[i] = np.sum(mask, axis = (0, 1)) / np.count_nonzero(mask, axis = (0, 1))[0]
      mask = np.zeros((img.shape), dtype = np.float32)

  imgfinal = np.zeros(img.shape, dtype = np.ndarray)

  rgb_centroids = normalize(centroids0, 0, 255).astype(np.uint8)

  att_min2 = att_min2.astype(np.uint8)


  for i in range(k):
    imgfinal[att_min2[:,:,0] == i] = rgb_centroids[i]

  imgfinal = imgfinal.astype(np.uint8)

  return imgfinal




I = input()

R = input()

op = int(input())

k = int(input())

n = int(input())

S = int(input())



'''Attributes initialization'''

img = iio.imread(I)

ref = iio.imread(R)

np.random.seed(S)



#As input cases are square images, the low parameter can be the img.shape[0]
centroids_pos = np.random.randint(img.shape[0], size = (k, 2)) 

rgb_centroids = np.zeros((k, 3), dtype = np.uint8)

for i in range(k):
  rgb_centroids[i] = img[centroids_pos[i, 0], centroids_pos[i, 1]]

rgbxy_centroids = np.concatenate((rgb_centroids, centroids_pos), axis = 1)
luminance_centroids = 0.299*rgb_centroids[:,0] + 0.587*rgb_centroids[:,1] + 0.114*rgb_centroids[:,2] 
luminancexy_centroids = np.concatenate((centroids_pos, np.reshape(luminance_centroids, (1, 5)).T), axis = 1)



imgfinal = kmeans(img, rgb_centroids, k, n)


print(RMSE(ref, imgfinal))

if op == 1:
  imgfinal = kmeans(img, rgb_centroids, k, n)
  print(RMSE(ref, imgfinal))

elif op == 2:
  imgfinal = kmeans(img, rgbxy_centroids, k, n)
  print(RMSE(ref, imgfinal))
  
elif op == 3:
  imgfinal = kmeans(img, luminance_centroids, k, n)
  print(RMSE(ref, imgfinal))
  
elif op == 4:
  imgfinal = kmeans(img, luminancexy_centroids, k, n)
  print(RMSE(ref, imgfinal))