#!/usr/bin/env python

import sys
import logging
import glob
import numpy as np
from PIL import Image
from skimage.transform import resize
import os
import re
import subprocess
import numpy
BIN = os.path.join(os.path.dirname(__file__), "jpegdir", "jpeg")

if not os.path.exists(BIN):
    print("jpeg is not built yet; use 'cd jpegdir; make' first")
    sys.exit(0)

def read (path):
    PATTERN = re.compile('\sC:(\d+)\s+N:(\S+)\s+W:(\d+)\s+H:(\d+)\s')
    BIN = os.path.join(os.path.dirname(__file__), "jpegdir", "jpeg")
    cmd = '%s -d -s %s' % (BIN, path)
    l = subprocess.check_output(cmd, shell=True)
    l = str(l)
    #print l
    m = re.search(PATTERN, l)
    C = int(m.group(1)) # I suppose this is # channels
    F = m.group(2)
    W = int(m.group(3))
    H = int(m.group(4))
    assert C == 1
    im = numpy.fromfile(F, dtype='uint16').reshape(H, W)
    L = im >> 8
    H = im & 0xFF
    im = (H << 8) | L
    os.remove(F)
    return im

def AllFindFiles():
    ListLJPEGPath = []
    for root, dirs, files in os.walk("/home/inchanhwang/ljpeg/normals"):
        for file in files:
            if file.endswith(".LJPEG"):
                apath = os.path.join(root, file)
                ListLJPEGPath.append(apath)
    return ListLJPEGPath

if __name__ == '__main__':
    ListLJPEGPath = AllFindFiles()
    for LJPEGFile in ListLJPEGPath:
        root = os.path.dirname(LJPEGFile)
        stem = os.path.splitext(LJPEGFile)[0]

        # read ICS
        ics = glob.glob(root + '/*.ics')[0]
        name = LJPEGFile.split('.')[-2]
        W = None
        H = None
        # find the shape of image
        for l in open(ics, 'r'):
            l = l.strip().split(' ')
            if len(l) < 7:
                continue
            if l[0] == name:
                W = int(l[4])
                H = int(l[2])
                bps = int(l[6])
                if bps != 12:
                    logging.warning('BPS != 12: %s' % LJPEGFile)
                break

        assert W != None
        assert H != None

        try:
            image = read(LJPEGFile)
        except:
            print("Unable to read : " + str(LJPEGFile))
            continue

        if W != image.shape[1]:
            logging.warning('reshape: %s' % LJPEGFile)
            image = image.reshape((H, W))

        raw = image
        rows, cols = image.shape
        im_dim_x = 800
        im_dim_y = 600

        print(LJPEGFile + "'s resolution is " + str(image.shape))

        im = resize(image, (im_dim_x, im_dim_y), anti_aliasing=True)

        rescaled_image = (np.maximum(im, 0) / im.max()) * 65535

        final_image = np.uint16(rescaled_image)
        print(LJPEGFile + "'s resolution rescaled to " + str(final_image.shape))

        final_image = Image.fromarray(final_image)
        NewNegativePath = './negatives/'
        if not os.path.exists(NewNegativePath):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs(NewNegativePath)
            print("Negative path is created!")

        file_name = os.path.basename(LJPEGFile)
        DstFolderPath = NewNegativePath + file_name + '.png'
        final_image.save(DstFolderPath)


