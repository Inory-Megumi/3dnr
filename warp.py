from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
import pyflow
import cv2

# Flow Options:
alpha = 0.012
ratio = 0.5
minWidth = 120
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

# read frame
import os,sys
img_folder = 'data/'
dirs = os.listdir(img_folder)
# sort files
dirs.sort(key = lambda x : int(x[1:x.find("_")]))


### following work ########
## 1. read ref and cur image and preprocessing
## 2. compute flow i0 -> i1
## 3. warp image i0
## 4. fuse warped i0 and i1

### 1 ######################
## prepare ref image
ref = np.array(Image.open(img_folder + dirs[0]))
# ref = np.array(Image.open('examples/gt/frame1.png'))
ref = ref.astype(float) / 255. 
################# 

#### 2 ,3 ,4######################

## flow calculation function
def flow(ref ,cur):
    u, v, warped = pyflow.coarse2fine_flow(
    cur, ref, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
    return u,v, warped

## warp function
def warp(input, flow):
    h, w = flow.shape[:2]
    warp_grid_x, warp_grid_y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    flow_inv = flow + np.stack((warp_grid_x, warp_grid_y), axis=-1)
    flow_inv = flow_inv.astype(np.float32)
    warped = cv2.remap(input, flow_inv, None, cv2.INTER_LINEAR)
    return warped

## flow visualization
def flow_vis(flow,shape):
    hsv = np.zeros(shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('flow.png', rgb)

## image fusion
def fuse(cur,warped):



    return 
######################

n = len(dirs)
for i in range(0,1):
    #cur = np.array(Image.open('examples/gt/frame2.png'))
    cur = np.array(Image.open(img_folder + dirs[i + 1]))
    cur = cur.astype(float) / 255.
    # u, v, im2W = pyflow.coarse2fine_flow(
    # ref, cur, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    #  nSORIterations, colType)
    u,v,warped = flow(ref, cur)
    ## optical flow visualization
    # flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    # flow_vis(flow,cur.shape)
    # warped = warp(ref, -flow)
    # cv2.imwrite('out/'+dirs[i+1] , warped[:, :, ::-1] * 255)
    # cv2.imwrite('warped.png', warped[:, :, ::-1] * 255)
    ## warped image and current image fusion
    ref = fuse(cur, warped)
    cv2.imwrite('out/'+dirs[i+1] , ref[:, :, ::-1] * 255)
# s = time.time()
# u, v, im2W = pyflow.coarse2fine_flow(
#     im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
#     nSORIterations, colType)
# e = time.time()
# print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
#     e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
# flow = np.concatenate((u[..., None], v[..., None]), axis=2)
#np.save('examples/outFlow.npy', flow)