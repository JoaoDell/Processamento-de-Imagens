''' ASSIGNMENT 1 - IMAGE GENERATION
    João Victor Dell Agli Floriano - 10799783
    SCC0251 Processamento de Imagens, 2022/1 
    Python 3.9.12 '''
import numpy as np
import random as rd


#Variable init
C = 0
Q = 1

#FUNCTIONS
def f1():
  '''f(x,y) = x * i + 2 * j'''
  img0 = np.zeros((C, C))

  for i in range(C):
    for j in range(C):
      img0[i][j] = i*j + 2*j

  return img0

def f2():
  '''f(x,y) = | cos( x/Q ) + 2 * sin( y/Q ) |'''
  img0 = np.zeros((C, C))

  for i in range(C):
    for j in range(C):
      img0[i][j] = np.absolute(np.cos(i/Q) + 2*np.sin(j/Q))

  return img0

def f3():
  '''f(x,y) = | 3 * ( x/Q ) - ( y/Q )**1/3 |'''
  img0 = np.zeros((C, C))

  for i in range(C):
    for j in range(C):
      img0[i][j] = np.absolute(3*(i/Q) - (j/Q)**(1/3))

  return img0

def f4():
  '''f(x,y) = random()'''
  img0 = np.zeros((C, C))

  for i in range(C):
    for j in range(C):
      img0[i][j] = rd.random()

  return img0

def f5():
  '''f(x,y) = randomwalks()'''
  img0 = np.zeros((C, C))

  x0 = 0
  y0 = 0
  img0[x0][y0] = 1

  for i in range(C):
    for j in range(C):

      dx = rd.randint(-1, 1)
      dy = rd.randint(-1, 1)

      x0 = (x0 + dx) % C
      y0 = (y0 + dy) % C

      img0[x0][y0] = 1
  
  return img0


# Conversion functions
def bit16_convert(img):
  '''Function that converts an image to 16-bit range.'''
  return ((img - np.min(img))/np.max(img))*(2**16 - 1.0)

def bit8_convert(img):
  '''Function that converts an image to 8-bit range.'''
  return ((img - np.min(img))/np.max(img))*(2**8 - 1.0)


# Downsampling function
def downsampling(img, Csize, Nsize):
  '''Function that converts an image to a lower resolution through one downsampling method.
     This method copies pixels from the original image (img) skipping k pixels,
     with k being the integer division between the original image lateral side (Csize) and the
     new image desired lateral side (Nsize)'''
  img0 = np.zeros((Nsize, Nsize))

  step = int(Csize/Nsize)

  for i in range(Nsize):
    for j in range(Nsize):
      img0[i][j] = img[step*i][step*j]

  return img0



#1 - PARAMETER INPUT
im_path = str(input("Reference image path: ")).rstrip()

C = int(input("Lateral size of the scene: "))
assert type(C) == int, "Lateral size must be an integer!\n"

func = int(input("Function to be used: "))
assert func in (1, 2, 3, 4, 5), "Function to be used must be an integer bewteen 1 and 5!\n"

Q = int(input("Parameter Q: "))
assert type(Q) == int, "Q must be an integer!\n"

N = int(input("Lateral size of the digital image: "))
assert N <= C, "Digital image lateral size must be lower or equal to C\n"

B = int(input("Number of bits per pixel: "))
assert 1 <= B <= 8, "Number of bits must be between 1 and 8\n"

seed = int(input("Seed for the random function: "))
assert type(seed) == int, "Seed must be an integer!\n"



#2 - GENERATING IMAGE SCENE
#dictionary with function headers to ease execution
func_dict = {1 : f1, 2 : f2, 3 : f3, 4 : f4, 5 : f5} 

f = np.zeros((C, C))
g = np.zeros((N, N))

rd.seed(seed)

#scene generation
f = func_dict[func]()
f = bit16_convert(f)



#3 - GENERATING DIGITAL IMAGE
# Digital image generation - downsampling and quantisation
g = downsampling(f, C, N)
g = bit8_convert(g)
g = g.astype(np.uint8)
g = g >> 8 - B



#4 - COMPARISON
R = np.load(im_path)
RSE = np.sqrt(np.sum((g-R)**2))



#5 - ROOT MEAN SQUARED ERROR BETWEEN F AND G PRINT
print('%0.4f' % RSE)