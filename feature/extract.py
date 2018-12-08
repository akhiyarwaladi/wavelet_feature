# extract.py
import os
import sys
import time
import datetime
import socket
import yaml
import shutil

sys.path.append('../utility')
import data_util as dutil

def main(argv):
    if len(argv)!=6:
        print ('USAGE:')
        print ('python extract.py [feature] [configPath] [dataset] [inDir] [outDir]')
        return

    feaName = argv[1]
    cfgFpath = argv[2]
    datasetName = argv[3]
    inDir = os.path.join(argv[4],datasetName)
    outDir = argv[5]

    print (feaName)

    extractor = None
    if feaName=='slbp':
        sys.path.append('./lbp/slbp'); import slbp
        extractor = slbp
    elif feaName=='wtio':
        sys.path.append('./wavelet/wtio'); import wtio
        extractor = wtio
    elif feaName== 'wmiftah':
        sys.path.append('./wavelet/wmiftah'); import wmiftah
        extractor = wmiftah
    elif feaName== 'haralick':
        sys.path.append('./glcm/haralick'); import haralick
        extractor = haralick
    elif feaName== 'shog':
        sys.path.append('./hog/shog'); import shog
        extractor = shog
    else:
        print ('FATAL: unknown featureName')
        return

    tag = '-'.join(['extract',feaName,datasetName,dutil.tag()])
    outDir = os.path.join(outDir,tag)
    if os.path.exists(outDir): shutil.rmtree(outDir)
    os.makedirs(outDir)
    shutil.copyfile(cfgFpath,os.path.join(outDir,feaName+'.cfg'))
    outDir = os.path.join(outDir,datasetName)
    os.makedirs(outDir)
    with open(cfgFpath,'r') as f: params = yaml.load(f)

    classList = os.listdir(inDir)
    for c in classList:
        if c ==".gitignore":
            continue
        cInDir = os.path.join(inDir,c)
        imgFnames = [f for f in os.listdir(cInDir)
                     if os.path.isfile(os.path.join(cInDir,f))
                        and (f.endswith(".jpg") or f.endswith('.JPG') or f.endswith('.bmp'))]
        nImg = len(imgFnames)

        cOutDir = os.path.join(outDir,c)
        if os.path.exists(cOutDir): shutil.rmtree(cOutDir)
        os.makedirs(cOutDir)

        for i,imgFname in enumerate(imgFnames):
            imgFpath = os.path.join(cInDir,imgFname)
            print ('extracting '+imgFpath+' '+str(i+1)+'/'+str(nImg))

            extractResult = extractor.extract(imgFpath,params,
                                              cOutDir,imgFname[0:-4])

if __name__ == '__main__':
    tic = time.time()
    main(sys.argv)
    print("--- %s seconds ---" % (time.time() - tic))
