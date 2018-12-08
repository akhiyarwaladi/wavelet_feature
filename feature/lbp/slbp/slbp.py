# slbp.py
import os
import sys

import skimage.io as io
import skimage.feature as fea

import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def extract(imgFpath,params,outdir,tag):
    img = cv2.imread(imgFpath,0)
    if img is None:

        img = io.imread(imgFpath,as_grey=True)
    lbpImg = fea.local_binary_pattern(img,
                                   P=params['nNeighbors'],
                                   R=params['radius'],
                                   method=params['method'])

    # http://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    # given the number of points p in the LBP,
    # there are p + 1 uniform patterns.
    # The final dimensionality of the histogram is thus p + 2,
    # where the added entry tabulates all patterns that are not uniform.
    histBin = np.arange(0,params['nNeighbors']+2+1)
    histRange = (0,params['nNeighbors']+2)
    (lbpHist,_) = np.histogram(lbpImg.ravel(),
                               bins=histBin,range=histRange,density=True)

    # output
    #np.savetxt(os.path.join(outdir,tag+'_slbpImg.txt'),lbpImg,delimiter=',')
    #np.savetxt(os.path.join(outdir,tag+'_slbpHist.txt'),lbpHist,delimiter=',')
    with open(os.path.join(outdir,tag+'_slbpHist.pkl'),'wb') as f: joblib.dump(lbpHist,f)
    
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,
                                             figsize=(9,6))
    plt.gray()

    ax1.imshow(img); ax1.axis('off')
    ax2.imshow(lbpImg); ax2.axis('off')
    ax3.hist(lbpImg.ravel(),normed=True,bins=histBin,range=histRange)
    ax3.set_ylabel('Percentage')
    ax3.set_xlabel('LBP values (method:'+params['method']+')')

    #plt.savefig(os.path.join(outdir,tag+'_slbpPlot.png'),bbox_inches='tight')
    plt.close()
    
    return lbpHist

def load(xrawList,yList,dirpath):
    xList = []
    for i,xraw in enumerate(xrawList):
        pklPath = os.path.join(dirpath,yList[i],xraw[0:-4]+'_slbpHist.pkl')
        with open(pklPath,'r') as f:
            x = joblib.load(f)
            xList.append(x)
    return xList

def _test(argv):
    pass

if __name__ == '__main__':
    tic = time.time()
    _test(sys.argv)
    print("--- %s seconds ---" % (time.time() - tic))
