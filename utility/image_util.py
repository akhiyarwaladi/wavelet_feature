# image_util.py
import cv2
import os
import time
import sys
import shutil
import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold


def main(argv):
    if len(argv)<=4:
        print 'USAGE:'
        print 'python image_util.py [utility] [inDir] [outDir] [param1 param2 ... paramN]'
        print(len(argv))
        return

    utility = argv[1]
    inDir = argv[2]
    outDir = argv[3]
    params = argv[4:]

    if utility == 'resize':
        resize(params,inDir,outDir)
    elif utility == 'pca':
        doPCA(params,inDir,outDir)
    elif utility == 'divideImage':
    	divideImage(params,inDir,outDir)
    elif utility == 'synthImage':
    	synthImage(params, inDir, outDir)
    else:
    	assert False

def resize(params,inDir,outDir):
    assert len(params) == 2
    scale = float(params[0])
    dataset = params[1]

    outDir = os.path.join(outDir,dataset+'-resized-'+str(scale))
    if os.path.exists(outDir): shutil.rmtree(outDir)
    os.makedirs(outDir)

    inDir = os.path.join(inDir,dataset)
    classList = os.listdir(inDir)
    for c in classList:
        if c ==".gitignore":
            continue
        cInDir = os.path.join(inDir,c)
        imgFnames = [f for f in os.listdir(cInDir)
                     if os.path.isfile(os.path.join(cInDir,f))
                        and (f.endswith(".jpg") or f.endswith('.JPG'))]
        nImg = len(imgFnames)

        cOutDir = os.path.join(outDir,c)
        if os.path.exists(cOutDir): shutil.rmtree(cOutDir)
        os.makedirs(cOutDir)

        for i,imgFname in enumerate(imgFnames):
            imgFpath = os.path.join(cInDir,imgFname)
            print 'resize '+imgFpath+' '+str(i+1)+'/'+str(nImg)
            img = cv2.imread(imgFpath)
            res = cv2.resize(img,None,fx=scale,fy=scale,
            interpolation=cv2.INTER_AREA)
            outFpath = os.path.join(cOutDir,imgFname[0:-4]+".png")
            cv2.imwrite(outFpath,res);

def divideImage(params, inDir, outDir):
	assert len(params) == 2
	##divde the column and row by params[0]
	divideNumber = int(params[0])
	dataset = params[1]

	outDir = os.path.join(outDir,dataset+'-divide-'+str(divideNumber))
	if os.path.exists(outDir): shutil.rmtree(outDir)
	os.makedirs(outDir)

	inDir = os.path.join(inDir,dataset)
	classList = os.listdir(inDir)

	for c in classList:
		if c ==".gitignore":
			continue
		cInDir = os.path.join(inDir,c)
		imgFnames = [f for f in os.listdir(cInDir)
					 if os.path.isfile(os.path.join(cInDir,f))
						 and (f.endswith(".jpg") or f.endswith('.JPG'))]
		nImg = len(imgFnames)
		
		cOutDir = os.path.join(outDir,c)
		if os.path.exists(cOutDir): shutil.rmtree(cOutDir)
		os.makedirs(cOutDir)

		for i,imgFname in enumerate(imgFnames):
			imgFpath = os.path.join(cInDir,imgFname)
			print 'split '+imgFpath+' '+str(i+1)+'/'+str(nImg)

			img = cv2.imread(imgFpath)
			##get rows and cols image
			rows, cols, channel = img.shape

			##calculate coordinate of x incremental and y incremental
			x = int(round(float(cols)/float(divideNumber)))
			y = int(round(float(rows)/float(divideNumber)))

			##rows
			for i in range(0,divideNumber):
			##cols
				for j in range(0,divideNumber):
				##create ROI img[row:row_range, col:col_range]
					outp = img[i*y:(i+1)*y, j*x:(j+1)*x]
					outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_"+str(i)+"_"+str(j)+".jpg")
					cv2.imwrite(outFpath,outp)
	
def doPCA(params,inDir,outDir):
    assert len(params) == 2
    n_components = int(params[0])
    dataset = params[1]

    outDir = os.path.join(outDir,dataset+'-pca-'+str(n_components))
    if os.path.exists(outDir): shutil.rmtree(outDir)
    os.makedirs(outDir)

    inDir = os.path.join(inDir,dataset)
    classList = os.listdir(inDir)

    dataList = []
    for c in classList:
        cInDir = os.path.join(inDir,c)
        imgFnames = [f for f in os.listdir(cInDir)
                     if os.path.isfile(os.path.join(cInDir,f))
                        and (f.endswith(".pkl"))]
        nImg = len(imgFnames)

        for i,imgFname in enumerate(imgFnames):
            imgFpath = os.path.join(cInDir,imgFname)
            print 'Load data '+imgFpath+' '+str(i+1)+'/'+str(nImg)
            with open(imgFpath,'r') as f:
                x = joblib.load(f)
                dataList.append(np.ravel(x))
        
    print 'Processing PCA transformation.'

    pca = PCA(n_components)
    pca.fit(dataList)
    dataList = pca.transform(dataList)

    for c in classList:
        cInDir = os.path.join(inDir,c)
        imgFnames = [f for f in os.listdir(cInDir)
                     if os.path.isfile(os.path.join(cInDir,f))
                        and (f.endswith(".pkl"))]
        nImg = len(imgFnames)

        cOutDir = os.path.join(outDir,c)
        if os.path.exists(cOutDir): shutil.rmtree(cOutDir)
        os.makedirs(cOutDir)

        for i,imgFname in enumerate(imgFnames):
            imgFpath = os.path.join(cInDir,imgFname)
            print 'Write data '+imgFpath+' '+str(i+1)+'/'+str(nImg)
            outFpath = os.path.join(cOutDir,imgFname[0:-4]+".pkl")

            np.savetxt(os.path.join(cOutDir,imgFname[0:-4]+'.txt'),dataList[i],delimiter=',')
            with open(os.path.join(cOutDir,imgFname[0:-4]+'.pkl'),'w') as f: joblib.dump(dataList[i],f)


if __name__ == '__main__':
    tic = time.time()
    main(sys.argv)
    print("--- %s seconds ---" % (time.time() - tic))

