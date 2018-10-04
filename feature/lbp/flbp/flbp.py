import os
import sys
import shutil
import time
import datetime
import json
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def extract(img,n,r):
    texture = _flbp(img,nNeighbors=n,radius=r)
    return texture

def load(xrawList,yList,dirpath):
    x = []
    for i,xraw in enumerate(xrawList):
        pklPath = os.path.join(dirpath,yList[i],xraw[0:-4]+'_flbp.pkl')

        texture = None
        with open(pklPath,'r') as f:
            texture = joblib.load(f)

        if texture!=None:
            x.append(texture)
        else:
            assert False,'FATAL: flbp not found'

    return x

def _flbp(imgOri, nNeighbors, radius):
    img = cv2.cvtColor(imgOri,cv2.COLOR_BGR2GRAY)

    blok = 2*radius + 1
    F = 4
    flbpWidth = 1 + len(img[1]) - blok
    flbpHeight = 1 + len(img) - blok

    histClbp = [0.00] * int(math.pow(2, nNeighbors))
    for y in range(0, flbpHeight):
        for x in range(0, flbpWidth):
            white = 0
            for i in range(0, blok):
                for j in range(0, blok):
                    #check white pixel in local region
                    if img[y + i][x + j]==255:
                        white = white + 1

            #hitung sudut antarpixel tetangga
            angle = 2*math.pi/nNeighbors

            #posisi pixel pusat
            centerX = int(x) + int(blok / 2)
            centerY = int(y) + int(blok / 2)

            #menampung kodel lbp
            arl = [0]

            #menampung nilai clbp
            clbp = [1.00]

            for i in range(0, nNeighbors):
                posX = int(centerX)+int(round((radius + 0.1) * math.cos(i * angle)))
                posY = int(centerY)-int(round((radius + 0.1) * math.sin(i * angle)))

                #mencari delta Pi int
                deltaP = int(img[posY][posX]) - int(img[centerY][centerX])

                #fuzzy thresholding
                if deltaP >= F:
                    for p in range(0, len(arl)):
                        temp = int(arl[p])
                        temp = temp + int(math.pow(2, i))
                        arl[p] = temp

                if deltaP > -1 * F and deltaP < F:
                    #buat cabang baru
                    jum_arl = len(arl)
                    for p in range(0, jum_arl):
                        arl = arl + [0]
                        clbp = clbp + [1.00]

                    #hitung kode lbp
                    q = 0
                    for p in range(jum_arl, len(arl)):
                        temp = int(arl[q])
                        temp = temp + int(math.pow(2, i))
                        arl[p] = temp
                        q = q + 1

                    #hitung clbp m0
                    median = int(len(clbp)/2)
                    for r in range(0, median):
                        mf = float(clbp[r])
                        mf = mf * float(F - deltaP)/(2*F)
                        clbp[r] = mf

                    #hitung clbp m1
                    a = 0
                    for s in range(median, len(clbp)):
                        mf = float(clbp[a])
                        mf = mf * float(F + deltaP)/(2*F)
                        clbp[s] = mf
                        a = a + 1

            for i in range(0, len(arl)):
                lbpVal = int(arl[i])
                clbpval = float(clbp[i])
                histclbp = float(histClbp[lbpVal])
                histclbp += clbpval
                if white != blok*blok:
                    histClbp[lbpVal] = float(histclbp)
    return histClbp

def _main(argv):
    if len(argv)!=5:
        print 'USAGE:';
        print 'python flbp.py [nNeighbors] [radius] [inDir] [outDir]'

    nNeighbors = int(argv[1])
    radius = int(argv[2])
    inDir = argv[3]
    outDir = argv[4]

    param = dict()
    param['radius'] = radius
    param['nNeighbors'] = nNeighbors

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    tag = '-'.join(['flbp',hostname,timestamp])
    outDir = os.path.join(outDir,tag)
    paramFpath = os.path.join(outDir,'param.json')
    datasetName = inDir.split('/')[-1]
    outDir = os.path.join(outDir,datasetName)

    classList = os.listdir(inDir)
    for c in classList:
        cDir = os.path.join(inDir,c)
        imgFnames = [f for f in os.listdir(cDir)
                     if os.path.isfile(os.path.join(cDir,f))]

        cOutDir = os.path.join(outDir,c)
        if os.path.exists(cOutDir):
            shutil.rmtree(cOutDir)
        os.makedirs(cOutDir)

        for imgFname in imgFnames:
            imgFpath = os.path.join(cDir,imgFname)
            print 'extracting '+imgFpath

            img = cv2.imread(imgFpath)
            texture = extract(img,nNeighbors,radius)

            print img.shape
            print len(texture)

            fname  = imgFname[0:-4]+'_flbp'
            with open(os.path.join(cOutDir,fname+'.fea'), 'w') as f:
                for i in texture: f.write(str(i)+'\n')
            with open(os.path.join(cOutDir,fname+'.pkl'), 'w') as f:
                joblib.dump(texture,f)

            plt.clf()
            plt.figure()
            plt.hist(texture, bins=50, normed=False)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('FLBP Histogram of '+fname)
            plt.savefig(os.path.join(cOutDir,fname+'.png'), bbox_inches='tight')
            plt.close()

            break
        break

    with open(paramFpath,'w') as f:
        json.dump(param,f,sort_keys=True,indent=2)

if __name__ == '__main__':
    tic = time.time()
    _main(sys.argv)
    print("--- %s seconds ---" % (time.time() - tic))
