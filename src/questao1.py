import sys
import os
import argparse
import asyncio
import numpy as np
import cv2
from stereo import stereo

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Questão 1')
  parser.add_argument('-d', '--directory', metavar='directory', type=str, default='../datasets/cones', help='Diretório com as imagens fonte')
  args = parser.parse_args()

  directory = args.directory

  if not os.path.isdir(directory):
    print('O diretório informado não existe')
    exit()

  if not (os.path.exists(f'{directory}/im0.png') and os.path.exists(f'{directory}/im1.png')):
    print('O diretório não contém as imagens stereo')
    exit()

  imgL = cv2.imread(f'{directory}/im0.png')
  imgR = cv2.imread(f'{directory}/im1.png')

  disp = asyncio.run(stereo(imgL, imgR, 0))
  cv2.imshow("Left Image", imgL)
  cv2.imshow("Right Image", imgR)

  cv2.imshow("Disparity Map", np.uint8(disp))
  
  cv2.waitKey()
  cv2.destroyAllWindows()

