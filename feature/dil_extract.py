# extract.py
import os
import sys
import time
import datetime
import socket
import yaml
import shutil
import multiprocessing as mp

from scoop import futures
from multiprocessing import Pool

sys.path.append('../utility')
import data_util as dutil

sys.path.append('./wavelet/wdil')
import wdil

# params, outDir1, kernel, feaName, datasetName, inDir, cfgFpath, distinct
def kernelProc(parameters): 
    params = parameters['params']
    outDir1 = parameters['outDir1']
    kernel = parameters['kernel']
    feaName = parameters['feaName']
    datasetName = parameters['datasetName']
    inDir = parameters['inDir']
    cfgFpath = parameters['cfgFpath']
    distinct = parameters['distinct']

    tag = '-'.join(['extract',feaName, datasetName, dutil.tag(), kernel])
    
    outDir2 = os.path.join(outDir1,tag)
    if os.path.exists(outDir2): shutil.rmtree(outDir2)
    os.makedirs(outDir2)

    shutil.copyfile(cfgFpath,os.path.join(outDir2,feaName+'.cfg'))
    outDir2 = os.path.join(outDir2,datasetName)
    os.makedirs(outDir2)

    classList = os.listdir(inDir)
    for c in classList:
        if c ==".gitignore":
            continue
        cInDir = os.path.join(inDir,c)
        imgFnames = [f for f in os.listdir(cInDir)
                     if os.path.isfile(os.path.join(cInDir,f))
                        and (f.endswith(".jpg") or f.endswith('.JPG'))]
        nImg = len(imgFnames)

        cOutDir = os.path.join(outDir2,c)
        if os.path.exists(cOutDir): shutil.rmtree(cOutDir)
        os.makedirs(cOutDir)
        for i,imgFname in enumerate(imgFnames):
            imgFpath = os.path.join(cInDir,imgFname)
            print 'extracting '+imgFpath+' '+str(i+1)+'/'+str(nImg)

            extractResult = wdil.extract(imgFpath,params,kernel,cOutDir,imgFname[0:-4],distinct)

def main(argv):
    if len(argv)!=6:
        print 'USAGE:'
        print 'python extract.py [feature] [configPath] [dataset] [inDir] [outDir]'
        return

    feaName = argv[1]
    cfgFpath = argv[2]
    datasetName = argv[3]
    inDir = os.path.join(argv[4],datasetName)
    outDir = argv[5]

    print feaName

    with open(cfgFpath,'r') as f: params = yaml.load(f)

    p = Pool(processes=mp.cpu_count())

    jumlahIter = len(params['kernel'])*(params['level']-2)
    print "Jumlah iterasi sebanyak "+str(jumlahIter)
    datasets_with_params = []

    for distinct in range(1,params['level']+1):
        toc = time.time()
        level = 'level-'+str(distinct)
        outDir1 = os.path.join(outDir, level)
        if os.path.exists(outDir1): shutil.rmtree(outDir1)
        os.makedirs(outDir1)
        
        for it in range(0,len(params['kernel'])):
            datasets_with_params.append(dict(params=params, outDir1=outDir1, kernel=params['kernel'][it],
                feaName=feaName, datasetName=datasetName, inDir=inDir, cfgFpath=cfgFpath, distinct=distinct))
    
    extractProcess = p.map(kernelProc, datasets_with_params)

        # print("--- Level "+str(distinct)+"/"+str(params['level'])+" complete in %s minutes ---" % ((time.time() - toc)/60))

        # for it in range(0,len(params['kernel'])):
        #     tac= time.time()
        #     iterasi+=1
        #     datasets_with_params = dict(params=params, outDir1=outDir1, kernel=params['kernel'][it], 
        #         feaName=feaName, datasetName=datasetName, inDir=inDir, cfgFpath=cfgFpath, 
        #         distinct=distinct)
        #     kernelProc(datasets_with_params)
        #     print("--- Iteration "+str(iterasi)+"/"+str(len(params['kernel']))+" "+"(Level "+str(distinct)+"/"+str(params['level'])+") complete in %s minutes ---" % ((time.time() - tac)/60))
        # print("--- Level "+str(distinct)+"/"+str(params['level'])+" complete in %s minutes ---" % ((time.time() - toc)/60))

if __name__ == '__main__':
    tic = time.time()
    main(sys.argv)
    print("--- Total time needed is %s minutes ---" % ((time.time() - tic)/60.0))
