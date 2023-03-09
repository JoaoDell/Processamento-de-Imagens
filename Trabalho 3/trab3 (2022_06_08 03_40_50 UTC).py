''' João Victor Dell Agli Floriano - 10799783
    SCC0251 Processamento de Imagens, 2022/1'''
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio

#Functions used
def normalize(img, min, max):
  '''Function that converts an image to the given desired range.'''
  return ((img - np.min(img))/(np.max(img) - np.min(img)))*(max - min) + min

def RMSE(img, img1):
  '''Function that calculates the RMSE between img and img1.'''
  return np.sqrt((np.sum((img - img1)**2))/(float(img.shape[0]*img.shape[1])))


#Inputs
I = str(input()).rstrip()

M = str(input()).rstrip()

G = str(input()).rstrip()


#Images load
image_filter = iio.imread(M)
input_image = iio.imread(I)
ref_image = iio.imread(G)



#Normalizing the filter
image_filter = normalize(image_filter, 0, 1)

#Fourier transforming
input_fourier = np.fft.fft2(input_image)

#Shifting so the frequencies allign to the center of the image
input_fourier = np.fft.fftshift(input_fourier)

#Multiplying by the filter
input_transformed = np.multiply(input_fourier, image_filter)

#Inverting the shift
input_transformed = np.fft.ifftshift(input_transformed)

#Inverse fourier transforming to reconstruct the new image after the transformation
input_transformed = np.fft.ifft2(input_transformed)




#Conversion to uint8
input_transformed = normalize(np.real(input_transformed), 0, 2**8 - 1)
input_transformed = input_transformed.astype(np.uint8)



#Printing the RMSE calculus
print('%0.4f' % RMSE(input_transformed, ref_image))