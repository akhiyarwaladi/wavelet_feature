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
from plotData import _plotEntropy, _plotHistogram, _plotSpectrum, _plotCoeffs

def _binningData(data, binValue):
    binData = np.array(data, copy=True)
    dataSize = float(len(binData)); maxVal = np.amax(binData); minVal = np.amin(binData)
    if (maxVal-minVal) > 0:
        binData = np.around((binData-minVal)/(maxVal-minVal)*binValue)
    test = np.amax(binData)
    return binData

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

def _calcHomogeneity(data, probaData):
    homogeneity = 0.0
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            homogeneity += probaData[int(data[i][j])]/(1+abs(i-j))
    return homogeneity

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

def _waveletPacket(img, outdir, level, kernel, subband, plot):
    wp = pywt.WaveletPacket2D(img, kernel, 'symmetric', level)
    
    # set wavelet sub band decomposition
    if subband == 'll':
        increment = 'a'
    elif subband == 'lh':
        increment = 'h'
    elif subband == 'hl':
        increment = 'v'
    elif subband == 'hh':
        increment = 'd'
    elif subband == 'all':
        pass
    else:
        print ('wavelet index error subband must be \'ll\' \'lh\' \'hl\' \'hh\' or \'all\'')

    if subband == 'all':
        # create energies, entropies list
        energies = list(); entropies = list()
        
        # loop over level
        index = ['a','h','v','d']
        for n in range(1,level+1):
            # get sub band index
            temp = list(index)
            for i in temp:
                # pop index
                del index[0]

                # plot
                if plot == True:
                    # calculate power spectrum
                    powerSpectrum = np.array(np.abs(wp[i].data)**2, copy=True)
                    outdirTemp = outdir+'level_'+str(n)+'_subband_'+i
                    _plotCoeffs(wp[i].data, outdirTemp+'_coeffs.png')
                    _plotSpectrum(powerSpectrum, outdirTemp+'_powerspectrum.png')
                    _plotHistogram(powerSpectrum, outdirTemp+'_hist.png')
                
                # binning data
                binData = _binningData(wp[i].data,1000)
                histData = _calcHist(binData, 1000)
                probaData = _calcProbaHist(histData)
                
                # calculate energy and entropy
                energy = np.sum(np.abs(wp[i].data))
                entropy = _calcEntropy(probaData)

                # append energy and entropy to list
                energies.append(energy)
                entropies.append(entropy)
          
                # generate next level index 
                if n < level+1:
                    index.append(i+'a'); index.append(i+'h'); index.append(i+'v'); index.append(i+'d')

        # calculate total energy
        total_energy = np.sum(energies)

        # check sanity
        if total_energy != 0:
            energies = energies/np.sum(energies)
        
        # append list of energy and entropy to feature
        features = np.concatenate((energies, entropies), axis=0)
        return features
    else:
        pass
        features = list()
        index = ''
        for n in range(1,level+1):
            # pass
            entropy = dict()
            outdirTemp = outdir+'level_'+str(n)+'_subband_'+index
            index += increment
        return features

def _wtioPreprocess(image, resize):
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
    
    # split channel
    b,g,r = cv2.split(image)

    # subtract channel
    GminR = cv2.subtract(g,r)
    GminB = cv2.subtract(g,b)
    RminG = cv2.subtract(r,g)

    return GminR, GminB, RminG

def extract(imgFpath,params,outdir,tag):
    image = cv2.imread(imgFpath)
    GminR, GminB, RminG = _wtioPreprocess(image, params['resize'])

    features = list()
    if params['plot'] == True:
        # explode the path
        fname = imgFpath.split('/')
        fname = fname[-1]
        # explode the ext
        fname = fname.split('.')
        fname = fname[-2]
        # create plot directory
        plotOutdir = os.path.join(outdir,fname+'/')

        if not os.path.exists(plotOutdir):
            os.makedirs(plotOutdir)
        features.append(_waveletPacket(GminR, plotOutdir+'blueChannel_', params['level'], params['kernel'], params['subband'], params['plot']))
        features.append(_waveletPacket(GminB, plotOutdir+'greenChannel_', params['level'], params['kernel'], params['subband'], params['plot']))
        features.append(_waveletPacket(RminG, plotOutdir+'redChannel_', params['level'], params['kernel'], params['subband'], params['plot']))
    else:
        features.append(_waveletPacket(GminR, os.path.join(outdir,tag), params['level'], params['kernel'], params['subband'], params['plot']))
        features.append(_waveletPacket(GminB, os.path.join(outdir,tag), params['level'], params['kernel'], params['subband'], params['plot']))
        features.append(_waveletPacket(RminG, os.path.join(outdir,tag), params['level'], params['kernel'], params['subband'], params['plot']))
    #flatten the features list
    features = np.ravel(features)
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