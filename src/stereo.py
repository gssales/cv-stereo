import time
import math
import multiprocessing as mp
import numpy as np

def stereo(imgL, imgR, cost_fn, patch_size=5, max_shift=80):
  sy, sx, _ = imgL.shape  
  disp = np.zeros((sy, sx))

  first_time = time.time()
  last_time = time.time()
  le = sy-2*(patch_size//2)
  for y in range(patch_size//2, sy-patch_size//2):
    cur_time = time.time()
    for x in range(patch_size//2, sx-patch_size//2):
      disp[y,x] = matchFeature(imgL, imgR, y, x, patch_size, max_shift, cost_fn)[2]

    elapsed = cur_time - last_time
    last_time = cur_time
    seconds = int(elapsed*(le-y-(patch_size//2)))
    minutes = seconds // 60
    seconds = seconds - minutes*60
    progress = ((y-(patch_size//2))*80)//sy
    print(f'{"█"*progress}{"░"*(80-progress)}', f'{minutes} min {seconds} s  ', end='\r')

  mapImg(disp, 0, max_shift, 0, 255)
    
  elapsed = int(last_time - first_time)
  minutes = elapsed // 60
  seconds = elapsed - minutes*60
  print("█"*80, f'{minutes} min {seconds} s  ')
  return disp

async def stereoAsync(imgL, imgR, cost_fn, patch_size=5, max_shift=80):
  global last_time, n, elapsed, sy, le
  sy, sx, _ = imgL.shape  
  disp = np.zeros((sy, sx))

  if not cost_fn:
    return disp

  elapsed = 0
  first_time = time.time()
  last_time = time.time()
  n = 0
  le = (sy-patch_size)*(sx-patch_size)
  def appendResult(result):
    y, x, d = result
    disp[y, x] = d
    
    cur_time = time.time()
    global last_time, n, elapsed, sy, le
    n = n+1
    elapsed = (elapsed*(n-1) + cur_time - last_time)/n
    last_time = cur_time
    seconds = int(elapsed*(le-n))
    minutes = seconds // 60
    seconds = seconds - minutes*60
    progress = (n*80)//le
    print(f'{"█"*progress}{"░"*(80-progress)}', f'{minutes} min {seconds} s  ', end='\r')

  pool = mp.Pool(mp.cpu_count())
  for y in range(patch_size//2, sy-patch_size//2):
    for x in range(patch_size//2, sx-patch_size//2):
      pool.apply_async(matchFeature, args=(imgL, imgR, y, x, patch_size, max_shift, cost_fn), callback=appendResult)
  pool.close()
  pool.join()

  mapImg(disp, 0, max_shift, 0, 255)

  elapsed = int(last_time - first_time)
  minutes = elapsed // 60
  seconds = elapsed - minutes*60
  print("█"*80, f'{minutes} min {seconds} s  ')
  return disp

def matchFeature(imgL, imgR, y, x, patch_size, max_shift, cost_fn):
  sy, sx, _ = imgL.shape  

  y0 = y-patch_size//2
  y1 = y+patch_size//2+1
  Lx0 = x-patch_size//2
  Lx1 = x+patch_size//2+1

  min_d = max_shift
  min_cost = 1e10
  for d in range(max_shift):
    Rx0 = Lx0-d 
    if Lx0-d < 0:
      continue
    Rx1 = Lx1-d #if Lx1-d < else patch_size/2+1

    cost = cost_fn(imgL[y0:y1,Lx0:Lx1], imgR[y0:y1,Rx0:Rx1], 1-(d/max_shift))
    if cost < min_cost:
      min_cost = cost
      min_d = d
  return (y, x, min_d)

def mapImg(img, in_min, in_max, out_min, out_max):
  for l in range(len(img)):
    for c in range(len(img[l])):
      img[l][c] = ((img[l][c] - in_min) * (out_max - out_min)) // (in_max - in_min + out_min) 
  return img
