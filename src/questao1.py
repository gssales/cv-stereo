import sys
import os
import argparse
import asyncio
import numpy as np
import cv2
from stereo import stereo, stereoAsync
from criterios import functions as fn

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Questão 1')
  parser.add_argument('-d', '--directory', metavar='directory', type=str, default='./datasets/cones', help='Diretório com as imagens fonte')
  parser.add_argument('-a', '--async', action='store_true', dest='a', help='Execução assíncrona')
  parser.add_argument('-p', '--patch_size', metavar='patch_size', type=int, default=1, help='Tamanho da janela de matching')
  parser.add_argument('-m', '--max_shift', metavar='max_shift', type=int, default=255, help='Máximo de deslocamento para procura pela disparidade')
  parser.add_argument('-c', '--cost_fn', metavar='cost_fn', type=str, default='ssd', help='Função de custo para encontrar a disparidade')
  parser.add_argument('-o', '--output', metavar='output', type=str, default='disp.png', help='nome do arquivo de saída')
  args = parser.parse_args()

  directory = args.directory
  asyn = args.a
  patch_size = args.patch_size
  max_shift = args.max_shift
  cost_fn = fn[args.cost_fn]

  if not os.path.isdir(directory):
    print('O diretório informado não existe')
    exit()

  if not (os.path.exists(f'{directory}/im0.png') and os.path.exists(f'{directory}/im1.png')):
    print('O diretório não contém as imagens stereo')
    exit()

  imgL = cv2.imread(f'{directory}/im0.png')
  imgR = cv2.imread(f'{directory}/im1.png')

  disp = []
  if asyn:
    disp = asyncio.run(stereoAsync(imgL, imgR, cost_fn, patch_size, max_shift))
  else:
    disp = stereo(imgL, imgR, cost_fn, patch_size, max_shift)
  cv2.imshow("Left Image", imgL)
  cv2.imshow("Right Image", imgR)

  cv2.imshow(f"Disparity Map ({args.output})", np.uint8(disp))
  
  cv2.waitKey()
  cv2.destroyAllWindows()
  
  cv2.imwrite(args.output, np.uint8(disp))
  print("Resultado Salvo")

