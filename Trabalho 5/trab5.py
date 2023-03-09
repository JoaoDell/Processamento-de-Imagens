''' João Victor Dell Agli Floriano - 10799783
    SCC0251 Processamento de Imagens, 2022/1'''
import numpy as np
import imageio as iio

'''Inputs'''


index = int(input())

Q_0 = (str(input()).rstrip()).split(' ')
Q = (int(Q_0[0]), int(Q_0[1]))

F = int(input())

T = int(input())

B = int(input())

names = np.array([], dtype = str)
for i in range(B):
  names = np.append( names, str(input()).rstrip() )



'''Functions'''

def convert_to_grayscale(img : np.ndarray):
  '''Function that converts an image to grayscale with standard color weights.\n'''
  return 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]


def limiarization(img, T):

  img1 = np.zeros(img.shape)

  img1[img >= T] = 1

  return img1


def complement(img : np.ndarray):

  img1 = np.full(img.shape, 1).astype(np.uint8)

  img1 = np.bitwise_xor(img1, img)

  print(img1)

  return img1


def difference(img1 :  np.ndarray, img2 :  np.ndarray):
  assert img1.shape == img2.shape, "img1 and img2 must have the same shape!\n"

  return np.bitwise_xor(img1, img2)

def union(img1 :  np.ndarray, img2 :  np.ndarray):
  assert img1.shape == img2.shape, "img1 and img2 must have the same shape!\n"

  return np.bitwise_or(img1, img2)

def intersection(img1 :  np.ndarray, img2 :  np.ndarray):
  '''Intersection set operation that receives:\n
     img1 : first image to be intersected\n
     img2 : second image to be intersected.'''
     
  assert img1.shape == img2.shape, "img1 and img2 must have the same shape!\n"

  return np.bitwise_and(img1, img2)

def translation(img : np.ndarray, s : int, t : int):
  '''Translation set operation that receives:\n
     img : img to be translated\n
     s : x translation\n
     t : y translation'''
  
  img1 = np.pad(img, ((s if s > 0 else 0, 0 if s > 0 else abs(s)), 
                      (t if t > 0 else 0, 0 if t > 0 else abs(t))), "constant", constant_values = 0)

  img1 = img1[ 0 if s > 0 else abs(s): img.shape[0] if s > 0 else img1.shape[0], 
               0 if t > 0 else abs(t): img.shape[1] if t > 0 else img1.shape[1] ]

  return img1


def reflection(img : np.ndarray):

  return np.flip(img)


def erosion(img : np.ndarray, filter : np.ndarray):
  '''Erosion function that receives:\n
     img : image to be dilated\n
     filter : filter to be applicated to the image. The filter will be padded.'''

  offset = (filter.shape[0] // 2, filter.shape[1] // 2)

  img_padded = np.pad(img, offset, "constant", constant_values = 0)

  imgnew = np.zeros(img.shape)

  m, n = filter.shape
  M, N = img.shape

  for i in range(0, M):
      for j in range(0, N):
        if 0 not in np.bitwise_and(filter, img_padded[i : (i + m), j : (j + n)]):
          imgnew[i, j] = 1
        


  return imgnew

  
def dilation(img : np.ndarray, filter : np.ndarray):
  '''Dilation function that receives:\n
     img : image to be dilated\n
     filter : filter to be applicated to the image. The filter will be padded.'''

  offset = (filter.shape[0] // 2, filter.shape[1] // 2)

  img_padded = np.pad(img, offset, "constant", constant_values = 0)

  imgnew = np.zeros(img.shape)

  m, n = filter.shape
  M, N = img.shape

  for i in range(0, M):
      for j in range(0, N):
        if 1 in np.bitwise_and(filter, img_padded[i : (i + m), j : (j + n)]):
          imgnew[i, j] = 1
        


  return imgnew



def opening(img : np.ndarray, filter : np.ndarray):

  imgnew = erosion(img, filter).astype(np.uint8)

  imgnew = dilation(imgnew, filter)

  return imgnew


def closing(img : np.ndarray, filter : np.ndarray):

  imgnew = dilation(img, filter).astype(np.uint8)

  imgnew = erosion(imgnew, filter)

  return imgnew


'''Haralick functions'''

def co_ocurrence_matrix(mask : np.ndarray, Q : tuple):

 
  '''Tried creating a matrix with only the values that existed in the mask,\n
     but couldnt figure out how to store which element corresponded to which 
  '''

  matrix = np.zeros((256, 256))

  for i in range(np.abs(Q[0]) if Q[0] < 0 else 0, mask.shape[0] - Q[0] if Q[0] >= 0 else mask.shape[0]):
    for j in range(np.abs(Q[1]) if Q[1] < 0 else 0, mask.shape[1] - Q[1] if Q[1] >= 0 else mask.shape[1]):
      matrix[ mask[i, j], mask[i + Q[0], j + Q[1]]] += 1

  a = np.sum(matrix)

  return matrix / a


def haralick(co_matrix : np.ndarray):


  '''[ 0 - auto_correlation, 1 - contrast, 2 - dissimilarity, 3 - energy, 4 - entropy, 
     5 - homogeneity, 6 - inverse difference, 7 - maximum probability]'''

  hara_mask = np.zeros(8)

  M, N = co_matrix.shape
  I, J = np.ogrid[0: M, 0: N]


  hara_mask[0] = (I*J*co_matrix).sum()
  hara_mask[1] = ( np.power((I - J), 2)*co_matrix).sum()
  hara_mask[2] = (np.abs(I - J)*co_matrix).sum()
  hara_mask[3] = (np.power(co_matrix, 2)).sum()
  hara_mask[4] = (- np.multiply(co_matrix[co_matrix > 0], np.log(co_matrix[co_matrix > 0]) )).sum()
  hara_mask[5] = np.divide(co_matrix, 1 + np.power((I - J), 2) ).sum()
  hara_mask[6] = np.divide(co_matrix, 1 + np.abs(I - J) ).sum()


  hara_mask[7] = np.max(co_matrix)

  return hara_mask


def euclidian_distance(array1 : np.ndarray, array2 : np.ndarray):
  return np.sqrt( np.sum( np.square(array1 - array2) ) )






'''Image opening and grayscale convertion'''
img_list = np.ndarray(B, dtype = np.ndarray)
img_list_gray = np.ndarray(B, dtype = np.ndarray)

for i in range(B):
  img_list[i] = iio.imread(names[i])
  img_list_gray[i] = convert_to_grayscale(img_list[i]).astype(np.uint8)


'''Limiarization'''

img_limiar = np.ndarray(B, dtype = np.ndarray)

for i in range(B):
  img_limiar[i] = limiarization( img_list_gray[i], T).astype(np.uint8)


'''Morphology'''
lmorph_list = np.ndarray(B, dtype = np.ndarray)

for i in range(B):
  lmorph_list[i] = np.zeros( img_list_gray[i].shape, dtype = np.uint8)


# Square 3x3 filter
filter =  np.full((3, 3), 1).astype(np.uint8)

if F == 1:

  for i in range(B):
    lmorph_list[i] = opening(img_limiar[i], filter)

else:

  for i in range(B):
    lmorph_list[i] = closing(img_limiar[i], filter)


'''Mask creation'''
mask1_list = np.ndarray(B, dtype = np.ndarray)
mask2_list = np.ndarray(B, dtype = np.ndarray)

for i in range(B):
  mask1_list[i] = np.zeros(lmorph_list[i].shape).astype(np.uint8)
  mask2_list[i] = np.zeros(lmorph_list[i].shape).astype(np.uint8)

  #Mask filtering
  mask1_list[i][lmorph_list[i] == 0] = img_list_gray[i][lmorph_list[i] == 0]
  mask2_list[i][lmorph_list[i] == 1] = img_list_gray[i][lmorph_list[i] == 1]



'''Co-ocurrence matrix'''
co_matrix1_list = np.ndarray(B, dtype = np.ndarray)
co_matrix2_list = np.ndarray(B, dtype = np.ndarray)

for i in range(B):
  co_matrix1_list[i] = co_ocurrence_matrix(mask1_list[i], Q)
  co_matrix2_list[i] = co_ocurrence_matrix(mask2_list[i], Q)



'''Mask creation'''
mask1_descriptor_list = np.ndarray(B, dtype = np.ndarray)
mask2_descriptor_list = np.ndarray(B, dtype = np.ndarray)
all_descriptors_list = np.ndarray(B, dtype = np.ndarray)

for i in range(B):  
  mask1_descriptor_list[i] = haralick(co_matrix1_list[i])
  mask2_descriptor_list[i] = haralick(co_matrix2_list[i])
  all_descriptors_list[i] = np.concatenate((mask1_descriptor_list[i], mask2_descriptor_list[i]), axis = None)



euclidian_sim_list = np.zeros(B, dtype = np.float32)

for i in range(B):
  euclidian_sim_list[i] = euclidian_distance(all_descriptors_list[index], all_descriptors_list[i])

euclidian_sim_list = euclidian_sim_list / np.sqrt( np.sum( np.square(euclidian_sim_list )) )
euclidian_sim_list = 1 - euclidian_sim_list



dictionary = {}
for i in range(B):
  dictionary[ euclidian_sim_list[i] ] = names[i]


euclidian_sim_list = np.sort(euclidian_sim_list)
euclidian_sim_list = euclidian_sim_list[::-1]


'''Printing'''
print("Query: " + names[index])
print("Ranking:")
for i in range(B):
  print("(" + str(i) + ") " + dictionary[euclidian_sim_list[i]])