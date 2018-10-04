import os
import sys
import shutil
import time
import datetime
import socket
import json
import cv2
import numpy as np
import pywt
import csv
import operator
from sklearn.externals import joblib
# from plotData import _plotEntropy, _plotHistogram, _plotSpectrum, _plotCoeffs

def _getEntropy(data, binValue):
    entropyList = []
    for i in data:
        binData = np.array(i, copy=True)
        dataSize = float(len(binData)); maxVal = np.amax(binData); minVal = np.amin(binData)
        if (maxVal-minVal) > 0:
            binData = np.around((binData-minVal)/(maxVal-minVal)*binValue)
        binData = _calcHist(binData,binValue)
        binData = _calcProbaHist(binData)
        entropy = _calcEntropy(binData)
        entropyList.append(entropy)
    return entropyList

def _calcHist(data, maxHistRange):
    histData = np.array([0.0]*(maxHistRange+1))
    for i in np.ravel(data):
        histData[int(i)] += 1
    return histData

def _calcProbaHist(histData):
    histData = histData/np.sum(histData)
    return histData

def _calcEntropy(probaData):
    entropy = 0.0
    for p in probaData:
        if p > 0:
            entropy -= (p*np.log2(p))
    return entropy    

def _normalize(data):
    outp = np.array(np.ravel(data), copy=True)
    maxVal = np.amax(np.abs(outp))
    if maxVal > 0:
        for i in range(0,len(outp)):
            outp[i] = outp[i]/maxVal
    return outp

def _dumpToCSV(data, outdir):
    with open(outdir+'.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in data.items():
           writer.writerow([key, value])
def _calcEnergy(data):
    feature = []
    for i in data:
        energy = np.sum(np.abs(i))
        feature.append(energy)

    return feature

def _dwt(img, kernel , level):
    featurelist = []
    prevLL = None
    for i in range(0,level):
        coefflist = []
        target = None
        if i == 0:
            target = img
        else:
            target = prevLL
        coeff = pywt.dwt2(target,kernel)
        LL,(LH,HL,HH) = coeff
        coefflist.append(LL); coefflist.append(LH); coefflist.append(HL); coefflist.append(HH)
        energies = _calcEnergy(coefflist)
        # calculate total energy
        total_energy = np.sum(energies)
        # check sanity
        if total_energy != 0:
            energies = energies/np.sum(energies)

        entropies = _getEntropy(coefflist,1000)
        features = np.concatenate((energies, entropies), axis=0)
        featurelist.append(features)
        prevLL = LL

    return featurelist


def _wmiftahPreprocess(image, resize):
    # set orientation to landscape
    rows, cols, _ = image.shape
    if rows > cols:
        image = np.rot90(image)
    
    # resize image
    rows,cols, _ = image.shape
    if cols > resize:
        ratio = float(cols)/float(resize)
        newRows = int(round(float(rows)/ratio))
        newCols = resize
        image = cv2.resize(image, (newCols, newRows))
    
    # blur the image
    image = cv2.blur(image, (5,5))
    
    #ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret, image = cv2.threshold(image, 12, 255, cv2.THRESH_BINARY)
    # split channel
    b,g,r = cv2.split(image)

    # subtract channel
    GminR = cv2.subtract(g,r)
    GminB = cv2.subtract(g,b)
    RminG = cv2.subtract(r,g)

    return GminR, GminB, RminG

def extract(imgFpath,params,outdir,tag):
    image = cv2.imread(imgFpath)
    GminR, GminB, RminG = _wmiftahPreprocess(image, params['resize'])
    features = list()
    features.append(_dwt(GminR,params['kernel'],params['level']))
    features.append(_dwt(GminB,params['kernel'],params['level']))
    features.append(_dwt(RminG,params['kernel'],params['level']))
    #flatten the features list
    features = np.ravel(features)
    print(features)
    np.savetxt(os.path.join(outdir,tag+'_wtio.txt'),features,delimiter=',')
    with open(os.path.join(outdir,tag+'_wtio.pkl'),'wb') as f: joblib.dump(features,f)
    return features

def load(xrawList,yList,dirpath):
    xList = []
    for i,xraw in enumerate(xrawList):
        pklPath = os.path.join(dirpath,yList[i],xraw[0:-4]+'_wtio.pkl')
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