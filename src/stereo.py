import time
import math
import multiprocessing as mp
import numpy as np

def stereo(imgL, imgR, cost_fn, patch_size=5, max_shift=80, box_size=0, agg_fn=lambda x: 0):
  sy, sx, _ = imgL.shape
  disp = np.zeros((sy,sx))
  
  I0 = patch_size//2 + box_size//2
  Iy1 = sy - patch_size//2 - box_size//2
  Ix1 = sx - patch_size//2 - box_size//2

  first_time = time.time()
  last_time = time.time()

  for y in range(I0, Iy1):  # Iy1-I0
    for x in range(I0, Ix1):  # Iy1-I0 * Ix1-I0
      Py0 = y - patch_size//2
      Py1 = y + patch_size//2 +1
      Px0 = x - patch_size//2
      Px1 = x + patch_size//2 +1
      
      Sy0 = Py0 - box_size//2
      Sy1 = Py1 + box_size//2 +1
      Sx0 = Px0 - box_size//2 - max_shift
      Sx0 = Sx0 if Sx0 >= 0 else 0
      Sx1 = Px1 + box_size//2 +1

      disp[y,x] = match_feature(imgL[Py0:Py1, Px0:Px1], imgR[Sy0:Sy1, Sx0:Sx1], cost_fn, max_shift, box_size, agg_fn)

    cur_time = time.time()
    elapsed = cur_time - last_time
    last_time = cur_time
    seconds = int(elapsed*(Iy1 -y -2*I0))
    minutes = seconds // 60
    seconds = seconds - minutes*60
    progress = ((y -I0)*80)//(Iy1 -I0)
    print(f'{"█"*progress}{"░"*(80-progress)}', f'{minutes} min {seconds} s  ', end='\r')
      
  mapImg(disp, 0, max_shift, 0, 255)

  elapsed = int(last_time - first_time)
  minutes = elapsed // 60
  seconds = elapsed - minutes*60
  print("█"*80, f'{minutes} min {seconds} s  ')

  return disp

def match_feature(patch, section, cost_fn, max_shift, box_size, agg_fn):
  cost_cube = compute_cost_cube(patch, section, cost_fn) # patch_size
  # print(cost_cube[:4,:4])
  sy, sx = cost_cube.shape
  min_d = 0
  min_cost = 1e5
  if sy == 0:
    return 0
  for d in range(box_size//2, sx-box_size//2):
    cost = 1e15
    if box_size > 0:
      x0 = d-box_size//2
      x1 = d+box_size//2+1
      cost = agg_fn(cost_cube[:, x0:x1])
    else:
      cost = cost_cube[0,d]
    # print(cost, min_cost, box_size//2, sx-box_size//2)
    if cost < min_cost:
      min_cost = cost
      min_d = max_shift-d-box_size//2
  return min_d

def compute_cost_cube(patch, section, cost_fn):
  py, px, _ = patch.shape
  sy, sx, _ = section.shape
  cost_cube = np.zeros((sy - py, sx - px))
  for y in range(py//2, sy - py//2 -1):
    for x in range(px//2, sx - px//2 -1):
      y0 = y - py//2
      y1 = y + py//2 +1
      x0 = x - px//2
      x1 = x + px//2 + 1
      cost_cube[y -py//2, x -px//2] = cost_fn(patch, section[y0:y1, x0:x1])
  return cost_cube


def agg(img, box_size, agg_fn):
  sy, sx = img.shape
  agg = np.zeros((sy,sx))
  for y in range(box_size//2, sy-box_size//2):
    for x in range(box_size//2, sx-box_size//2):
      y0 = y-box_size//2
      y1 = y+box_size//2+1
      x0 = x-box_size//2
      x1 = x+box_size//2+1
      agg[y,x] = agg_fn(img[y0:y1, x0:x1])
  return agg

def avg(box):
  if np.size(box) == 0: 
    print(box, np.size(box))
    raise Exception()
  return np.sum(box)/np.size(box)
  
def median(box):
  box = box.flatten()
  box.sort()
  return box[np.size(box)//2]

def mapImg(img, in_min, in_max, out_min, out_max):
  for l in range(len(img)):
    for c in range(len(img[l])):
      img[l][c] = ((img[l][c] - in_min) * (out_max - out_min)) // (in_max - in_min + out_min) 
  return img

functions = {
  'avg': avg,
  'median': median
}
