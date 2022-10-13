import cv2
import numpy as np
import glob

frameSize = (3840, 1080)
# read frame
import os,sys
img_folder = 'out/noise'
dns_folder = 'out/old'
dirs = os.listdir(img_folder)
# sort files
dirs.sort(key = lambda x : int(x[1:x.find("_")]))
out = cv2.VideoWriter('out/video/output_comp_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 2, frameSize)
i = 0
for filename in dirs:
    if(i > 9):
        break
    img = cv2.imread(img_folder+'/'+filename)
    img_dns = cv2.imread(dns_folder+'/'+filename)
    img_com = np.hstack((img,img_dns))
    print(img_com.shape)
    out.write(img_com)
    i = i + 1
out.release()