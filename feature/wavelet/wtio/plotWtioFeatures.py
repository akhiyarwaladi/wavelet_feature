# image_util.py
import cv2
import os
import time
import sys
import shutil
import numpy as np
import math
import csv
from sklearn.externals import joblib
import matplotlib.pyplot as plt

def main(argv):
    if len(argv)<=4:
        print 'USAGE:'
        print 'python plotWtioFeatures.py [inDir] [outDir] [param1 param2 ... paramN]'
        print(len(argv))
        return
    
    inDir = argv[1]
    outDir = argv[2]
    params = argv[3:]

    plotWtioFeatures(params, inDir, outDir)

def plotWtioFeatures(params, inDir, outDir):
    #plotWtioFeatures is a function to plot wtio features
    assert len(params) == 2
    dataset = params[0]
    #set the background dir
    totalChannel = int(params[1])

    outDir = os.path.join(outDir,dataset+'-wtio-plot')
    if os.path.exists(outDir): shutil.rmtree(outDir)
    os.makedirs(outDir)

    inDir = os.path.join(inDir,dataset)
    classList = os.listdir(inDir)

    featuresList = []
    #read the features path
    for c in classList:
        cInDir = os.path.join(inDir,c)
        feaFnames = [f for f in os.listdir(cInDir)
                     if os.path.isfile(os.path.join(cInDir,f)) and (f.endswith(".pkl"))]
        nFea = len(feaFnames)
        
        cOutDir = os.path.join(outDir,c)
        if os.path.exists(cOutDir): shutil.rmtree(cOutDir)
        os.makedirs(cOutDir)
        #array to store all entropies and energies
        featuresList = []

        for i,feaFname in enumerate(feaFnames):
            feaFpath = os.path.join(cInDir,feaFname)
            print 'load '+feaFpath+' '+str(i+1)+'/'+str(nFea)
            #load feature data
            with open(feaFpath,'r') as f: features = joblib.load(feaFpath)
            featuresList.append(features)

        with open(cOutDir+'.csv','wb')as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(featuresList)

        #sum the features by column
        featuresList = np.sum(featuresList, axis = 0)
        featuresList = np.array(featuresList, dtype = float)
        #get the features mean
        featuresList /= nFea
        #get the level
        feaLenPerChannel = len(featuresList)/(2*totalChannel)
        temp = feaLenPerChannel
        level = 0
        while temp != 0:
            temp -= 4**(level+1)
            level += 1

        #find 2nd highest value energy to plot energy with scaling view
        '''
        energiesList = []
        shiftEnergiesListIndex = 0
        for i in range(0, totalChannel):
            feaLen = len(featuresList)
            energiesList.append(featuresList[(shiftEnergiesListIndex+i)*feaLen/(totalChannel*2):(shiftEnergiesListIndex+i+1)*feaLen/(totalChannel*2)])
            shiftEnergiesListIndex += 1

        energiesList = np.abs(np.ravel(energiesList))
        index = np.argsort(energiesList)
        ylimMax = energiesList[index[-totalChannel*level-1]]
        '''
        #slice feature per channel
        #i == 0 is equals to channel 1, i == 1 is equals to channel 2, i == n is equals to channel n+1
        for i in range(0, totalChannel):
            feaLen = len(featuresList)
            #slice array features to feature per channel
            #in wtio feature structure data features = [energies_channel_1 entropies_channel_1 energy_channel_2 energies_channel_2 ... ... ]    
            channelFeatures = featuresList[i*feaLen/totalChannel:(i+1)*feaLen/totalChannel]
            #slice channel features to channel Energies
            channelEnergies = channelFeatures[0:len(channelFeatures)/2]
            #slice channel features to channel Entropies
            channelEntropies = channelFeatures[len(channelFeatures)/2:len(channelFeatures)]
            #slice feature channel to feature channel per level
            shiftIndex = 0
            chEnergiesLevel = []
            chEntropiesLevel = []
            #j == 0 is equals to level 1, j == 1 is equals to level 2
            for j in range(0,level):
                print 'plotting channel ' + str(i+1) + ' level ' + str(j+1)
                if j == 0:
                    chEnergiesLevel = channelEnergies[0:4]
                    chEntropiesLevel = channelEntropies[0:4]
                    shiftIndex = 4
                else:
                    chEnergiesLevel = channelEnergies[shiftIndex:4**(j+1)+shiftIndex]
                    chEntropiesLevel = channelEntropies[shiftIndex:4**(j+1)+shiftIndex]
                    shiftIndex = shiftIndex+4**(j+1)

                ################### 1 Dimension graphics #################################
                #set ticker
                major_ticks = np.arange(0, 4**(j+1), 4)
                minor_ticks = np.arange(0, 4**(j+1), 1)
                ##plot energy
                ax1 = plt.subplot(2,1,1)
                plt.plot(chEnergiesLevel, marker='o', color='b')
                plt.title('Energy and Entropy channel ' + str(i+1) + ' wavelet packet level '+str(j+1))
                if j == 0:
                    plt.xlim(0,3)
                else:
                    plt.xlim(0,4**(j+1)-1)
                plt.ylim(-1,1)
                plt.ylabel('Absolute log energy')
                if j == 0:
                    plt.xticks(np.arange(4))
                else:
                    ax1.set_xticks(major_ticks)                                                       
                    ax1.set_xticks(minor_ticks, minor=True)
                plt.grid(which='both')  
                ax1.grid(which='minor', alpha=0.3)                                                
                ax1.grid(which='major', alpha=0.7)

                ax2 = plt.subplot(2,1,2)
                plt.plot(chEntropiesLevel, marker='o', color='b')
                if j == 0:
                    plt.xlim(0,3)
                else:
                    plt.xlim(0,4**(j+1)-1)
                plt.ylim(0,10)
                plt.xlabel('subband')
                plt.ylabel('Entropy')

                if j == 0:
                    plt.xticks(np.arange(4))
                else:
                    ax2.set_xticks(major_ticks)                                                       
                    ax2.set_xticks(minor_ticks, minor=True)
                plt.grid(which='both')
                ax2.grid(which='minor', alpha=0.3)                                                
                ax2.grid(which='major', alpha=0.7)
                plt.savefig(cOutDir+'_channel_'+str(i+1)+'_energies_entropies_mean_level_'+str(j+1)+'.jpg')
                plt.close()
                plt.close('all')
                plt.gcf().clear()

                ##################### plot scalling view #################################
                
                ##plot energy with scaling view
                ax1 = plt.subplot(2,1,1)

                plt.plot(np.log2(np.abs(chEnergiesLevel)), marker='o', color='b')
                plt.title('Energy (Absolute log Scalling) and entropy channel ' + str(i) + ' wavelet packet level '+str(j+1))
                if j == 0:
                    plt.xlim(0,3)
                else:
                    plt.xlim(0,4**(j+1)-1)
                ##plt.ylim(-ylimMax,ylimMax)
                plt.ylim(-50,0)
                plt.ylabel('Energy')
                if j == 0:
                    plt.xticks(np.arange(4))
                else:
                    ax1.set_xticks(major_ticks)                                                       
                    ax1.set_xticks(minor_ticks, minor=True)
                plt.grid(which='both')
                ax1.grid(which='minor', alpha=0.3)                                                
                ax1.grid(which='major', alpha=0.7)

                ax2 = plt.subplot(2,1,2)
                plt.plot(chEntropiesLevel, marker='o', color='b')
                if j == 0:
                    plt.xlim(0,3)
                else:
                    plt.xlim(0,4**(j+1)-1)
                plt.ylim(0,10)
                plt.xlabel('subband')
                plt.ylabel('Entropy')
                if j == 0:
                    plt.xticks(np.arange(4))
                else:
                    ax2.set_xticks(major_ticks)                                                       
                    ax2.set_xticks(minor_ticks, minor=True)
                plt.grid(which='both')
                ax2.grid(which='minor', alpha=0.3)                                                
                ax2.grid(which='major', alpha=0.7)
                plt.savefig(cOutDir+'_channel_'+str(i+1)+'_energies_entropies_mean_level_'+str(j+1)+'_scaled.jpg')
                plt.close()
                plt.close('all')
                plt.gcf().clear()


if __name__ == '__main__':
    tic = time.time()
    main(sys.argv)
    print("--- %s seconds ---" % (time.time() - tic))


