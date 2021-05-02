import numpy as np
  
def c1(patchL, patchR):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 1**2))
  
def c10(patchL, patchR):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 10**2))

def c50(patchL, patchR):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 50**2))

def c100(patchL, patchR):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(d / (d + 100**2))
  
def c500(patchL, patchR):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  d = patchL - patchR
  d = np.power(d, 2)
  return np.sum(np.sqrt(np.sum(d / (d + 500**2), 2)))

def ssd(patchL, patchR):
  patchL = np.int32(patchL)
  patchR = np.int32(patchR)
  return np.sum(np.sqrt(np.sum(np.power(patchL - patchR, 2), axis=2)))
  
# def ssd(patchL, patchR):
#   patchL = np.sqrt(np.sum(np.power(np.int32(patchL), 2), axis=2))
#   patchR = np.sqrt(np.sum(np.power(np.int32(patchR), 2), axis=2))
#   return np.sum(np.power(patchL - patchR, 2))

functions = {
    'c1': c1,
    'c10': c10,
    'c50': c50,
    'c100': c100,
    'c500': c500,
    'ssd': ssd
}