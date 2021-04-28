import time
import math
import multiprocessing as mp
  
async def stereo(imgL, imgR, patch_size):
  r = []

  global last_time, n, elapsed
  elapsed = 0
  last_time = time.time()
  n = 0
  def append(result):
    cur_time = time.time()
    r.append(result)
    global last_time, n, elapsed
    n = n+1
    elapsed = (elapsed*(n-1) + cur_time - last_time)/n
    last_time = cur_time
    progress = (len(r)*80)//len(imgL)
    print(f'{"█"*progress}{"░"*(80-progress)}', f'{int(elapsed*(len(imgL)-len(r)))} s  ', end='\r')


  pool = mp.Pool(mp.cpu_count())
  
  for l in range(len(imgL)):
    pool.apply_async(runLine, args=(imgL, imgR, l), callback=append)

  pool.close()
  pool.join()
  print()

  r.sort(key=lambda x: x[0])
  r = [l for i, l in r]

  max_value = -10000
  min_value = 10000
  for l in r:
    max_value = max(max_value, max(l))
    for c in l:
      min_value = min(min_value, c) if c > 0.0 else min_value
  
  for y in range(len(r)):
    for x in range(len(r[y])):
      r[y][x] = ((r[y][x] - min_value) * 255) / (max_value - min_value);
  
  return r

def runLine(imgL, imgR, l):
  line = []
  for p in range(len(imgL[l])):
    x = imgL[l][p]
    d = 0
    min_d = d
    min_cost = 100000
    while d <= max(p, p - len(imgL[l])/2):
      cost = squared_diff(imgL, imgR, l, p, -d, 0)
      last_cost = min_cost
      min_cost = min(cost, min_cost)
      if last_cost != min_cost:
        min_d = d
      d = d+1
    line.append(abs(min_d))
  return (l, line)

def squared_diff(IL, IR, y, x, d, patch_size):
  p = patch_size
  s = 0
  for i in range(3):
    c = int(IL[y][x+d][i]) - int(IR[y][x][i])
    s = s + c*c
  return s
