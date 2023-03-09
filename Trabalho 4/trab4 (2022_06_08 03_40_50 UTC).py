'''João Victor Dell Agli Floriano - 10799783
   SCC0251 Processamento de Imagens, 2022/1'''
import numpy as np
import matplotlib.pyplot as plt
import imageio as iio

def RMSE(img, img1):
  '''Function that calculates the RMSE between img and img1.'''
  return np.sqrt((np.sum((img - img1)**2))/(float(img.shape[0]*img.shape[1])))

def normalize(img : np.ndarray, min = 0.0, max = 1.0):
  '''Function that converts an image to the given desired range.'''
  return ((img - np.min(img))/(np.max(img) - np.min(img)))*(max - min) + min

def gaussian_filter(k: int, sigma: float):
  '''Returns a gaussian filter from the following parameters:
     - k : int -> lateral size of the filter
     - sigma : float -> size of the blur. The higher the sigma the lower the blurred area, hence weaker the blurring effect.
  '''
     
  rangex = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
  x, y = np.meshgrid(rangex, rangex)
  filt = np.exp(-(1/2)*(np.square(x) + np.square(y)) / np.square(sigma)) 
  return filt/np.sum(filt)


def angle_PSF(x: int, y: int, angle: float, size: int):
  '''Returns a PSF filter that simulates motion blur from the following parameters:
     - x : int -> horizontal size of the filter
     - y : int -> vertical size of the filter
     - angle : float -> angle of inclination of the motion
     - size : int -> distance of the motion blur. The higher the distance, the more blurred the image. 
  '''

  PSF = np.zeros((x, y))
  center = np.array([x-1, y-1]) // 2
  radians = np.radians(angle)
  
  phase = np.array([np.cos(radians), np.sin(radians)])

  for i in range(size):
      offset_x = int(center[0] - np.round_(i*phase[0]))
      offset_y = int(center[1] - np.round_(i*phase[1]))
      PSF[offset_x, offset_y] = 1 

  PSF /= PSF.sum()

  return PSF



def const_least_squares(g : np.ndarray, h : np.ndarray, gamma : float):
  '''Returns the constrained least-squares restoration of an image. It has the following parameters:
     - g : np.ndarray -> Original distorted image
     - h : np.ndarray -> Image filter
     - gamma : float -> Regularization parameter. Must be inside the range [0, 1) 
  '''
  F = np.zeros(g.shape)

  #laplacian operator
  p = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]], dtype = "float64")

  #padding the laplacian operator and the filter
  a = ( ( (g.shape[0] - p.shape[0]) // 2, (g.shape[1] - p.shape[1]) // 2 + (1 - p.shape[0] % 2)), 
        ( (g.shape[0] - p.shape[0]) // 2, (g.shape[1] - p.shape[1]) // 2 + (1 - p.shape[1] % 2) ) )
  
  b = ( ( (g.shape[0] - h.shape[0]) // 2, (g.shape[1] - h.shape[1]) // 2 + (1 - h.shape[0] % 2)), 
        ( (g.shape[0] - h.shape[0]) // 2, (g.shape[1] - h.shape[1]) // 2 + (1 - h.shape[1] % 2) ) )

  p_padded = np.pad(p, a, "constant", constant_values = 0)
  h_padded = np.pad(h, b, "constant", constant_values = 0)


  #Converting everything to the fourier space
  H = np.fft.fft2(h_padded)
  G = np.fft.fft2(g)
  P = np.fft.fft2(p_padded)

  H_star = np.conjugate( H )

  #Calculating the F(x,y) 
  div = np.square( np.abs(H) ) + gamma * np.square( np.abs(P) )
  div = np.divide(H_star, div)
  F = np.multiply(div, G)

  #Converting back to the original image space
  f = np.fft.fftshift(np.fft.ifft2(F).real)
  f = np.clip(f, 0, 255).astype(np.uint8)

  return f 


def richardson_lucy(g, h, steps):
  
  #Initializing the variables
  R_new = np.full(shape = img.shape, fill_value = 1, dtype = "float64")

  #ocnverting the filter to the fourier space
  PSF = np.fft.fft2(h)
  PSF_T = np.transpose( np.conjugate( PSF ) )
 
  for i in range(steps):
    #fourier space operations (convolution)
    div = np.multiply(np.fft.fft2(R_new), PSF)

    #Back to normal space to divide
    div = np.fft.fftshift(np.fft.ifft2(div).real)
    #Restricting lower values so there is now division by zero
    div[ div < 0.1 ] = 0.0001

    div = np.divide(g, div)

    #Back to fourier space to convolve again
    div = np.fft.fft2(div)
    div = np.multiply(div, PSF_T)

    #Again back to normal space to do a point-wise multiplication seu gostoso
    R_new = np.multiply(R_new, np.fft.fftshift( np.abs( np.fft.ifft2( div ) ) ) )
    # R_new = normalize(R_new, 0.0, 1.0)

  R_new = normalize(R_new, 0, 255).astype(np.uint8)  

  return R_new




#Richardson-Lucy
angle = 0
steps = 0

#Constrained least-squares Filter
k = 0
sigma = 0.0
gamma = 0

I = (input()).rstrip()

method = int(input())
assert method in (1, 2), "Method input must be:\n- 1 - Richardson-Lucy\n- 2 - Constrained least-squares Filter"

img = iio.imread(I)

if method == 2:
  angle = float(input())

  steps = int(input())

  h = angle_PSF(img.shape[0], img.shape[1], angle, 20)

  f = richardson_lucy(img, h, steps)

  ''' Comparison '''
  print('%0.4f' % RMSE(img, f))

else:
  k = int(input())

  sigma = float(input())

  gamma = float(input())
  assert 0 <= gamma < 1, "Gamma value must be in higher or equal to 0 and lower than 1!"


  ''' Constrained least-squares '''
  img = iio.imread(I)

  h = gaussian_filter(k, sigma)

  f = const_least_squares(img, h, gamma)
  f = normalize(f, 0, 255)

  ''' Comparison '''
  print('%0.4f' % RMSE(img, f))