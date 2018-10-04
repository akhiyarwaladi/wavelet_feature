from matplotlib import gridspec
from sklearn.externals import joblib
import operator
import matplotlib.pyplot as plt
import numpy as np
import copy

def _plotEntropy(data, outdir):
    plt.bar(range(len(data)), data.values(), align='center')
    plt.xticks(range(len(data)), data.keys())
    plt.savefig(outdir)
    plt.close()
    plt.close('all')
    plt.gcf().clear()

def _plotHistogram(data, outdir):
    outp = np.array(np.ravel(data), copy=True)

    maxVal = np.amax(outp)
    outp = outp.astype(int)
    ##plot hist
    bins = np.arange(0,int(maxVal),int(float(maxVal)*0.01))

    plt.hist(outp, bins=bins, alpha=0.5)
    plt.title('Power spectrum histogram')
    plt.xlabel('Power spectrum')
    plt.ylabel('Frequency')
    plt.ylim(0,2000)
    plt.savefig(outdir)
    plt.close()
    plt.close('all')
    plt.gcf().clear()

def _plotSpectrum(data, outdir):
    rows = len(data)
    cols = len(data[0])
    
    fig = plt.figure(figsize=(10, 6))
    ##### Set plot view size #######
    fig.set_size_inches(12.8, 7.2)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ##### Power spectrum ###########
    ax2 = plt.subplot(gs[0])
    ax2.plot(np.ravel(data))
    plt.title('Power Spectrum')
    plt.xlim(0,rows*cols)
    plt.xlabel('Position')
    plt.ylabel('Power Spectrum')
    ##### Contour ##############
    ax1 = plt.subplot(gs[1])
    ax1.contourf(data, cmap=plt.cm.jet)
    plt.title('map = contourf, ' + 'dim = '+ str(cols) + 'x' + str(rows))
    plt.tight_layout()
    plt.savefig(outdir)
    plt.close()
    plt.close('all')
    plt.gcf().clear()

def _plotCoeffs(data, outdir):
    rows = len(data)
    cols = len(data[0])
    
    fig = plt.figure(figsize=(10, 6))
    ##### Set plot view size #######
    fig.set_size_inches(12.8, 7.2)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ##### Power spectrum ###########
    ax2 = plt.subplot(gs[0])
    ax2.plot(np.ravel(data))
    plt.title('Wavelet coeff')
    plt.xlim(0,rows*cols)
    plt.xlabel('Position')
    plt.ylabel('Frequency')
    ##### Contour ##############
    ax1 = plt.subplot(gs[1])
    ax1.imshow(data, cmap='gray')
    plt.title('map = gray, ' + 'dim = '+ str(cols) + 'x' + str(rows))
    plt.tight_layout()
    plt.savefig(outdir)
    plt.close()
    plt.close('all')
    plt.gcf().clear()