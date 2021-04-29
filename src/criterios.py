import numpy as np

def ssd_distance(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  v = np.sum(np.power(patchL - patchR, 2))
  return v / distance**3
  
def c100d(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  v = c100(patchL, patchR, distance)
  return v / distance

def c05(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 0.5**2))
  
def c1(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 1**2))
  
def c5(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 5**2))

def c10(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 10**2))

def c50(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 50**2))

def c100(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 100**2))
  
def c500(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 500**2))

def ssd(patchL, patchR, distance):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  return np.sum(np.power(patchL - patchR, 2))

functions = {
    'c100d': c100d,
    'c1': c1,
    'c10': c10,
    'c50': c50,
    'c100': c100,
    'c500': c500,
    'ssd': ssd
}