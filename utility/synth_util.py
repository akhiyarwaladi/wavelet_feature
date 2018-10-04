# image_util.py
import cv2
import os
import time
import sys
import shutil
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from random import randint

def main(argv):
    if len(argv)<=4:
        print 'USAGE:'
        print 'python synth_util.py [utility] [inDir] [outDir] [param1 param2 ... paramN]'
        print(len(argv))
        return

    utility = argv[1]
    inDir = argv[2]
    outDir = argv[3]
    params = argv[4:]

    if utility == 'synthBackground':
    	synthBackground(params, inDir, outDir)
    elif utility == 'synthNoise':
    	synthNoise(params, inDir, outDir)
    elif utility == 'blur':
    	synthBlur(params, inDir, outDir)
    else:
    	assert False

def synthBackground(params, inDir, outDir):
	#synthImage is a function to synthetic your image with another background
	assert len(params) == 4
	dataset = params[0]
	#set the background dir
	backgroundDir = params[1]
	#set the threshold size of your image
	threshValue = int(params[2])
	#set the kernel size of erode and dilate operation
	morphKSize = int(params[3])
	
	outDir = os.path.join(outDir,dataset+'-synthBG')
	if os.path.exists(outDir): shutil.rmtree(outDir)
	os.makedirs(outDir)

	inDir = os.path.join(inDir,dataset)
	classList = os.listdir(inDir)

	#read background path
	backgroundClassList = os.listdir(backgroundDir)
	#variable to store background dir list
	backgroundList = []

	for c in backgroundClassList:
		if c == ".gitignore":
			continue
		cInDir = os.path.join(backgroundDir, c)
		imgFnames = [f for f in os.listdir(cInDir)
					 if os.path.isfile(os.path.join(cInDir,f))
					 	and (f.endswith(".jpg") or f.endswith (".JPG"))]
		nImg = len(imgFnames)
		#append background dir list
		for i,imgFname in enumerate(imgFnames):
			imgFpath = os.path.join(cInDir,imgFname)
			backgroundList.append(imgFpath)
	#read the image path
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
			nBackground = len(backgroundList)
			imgFpath = os.path.join(cInDir,imgFname)
			print 'synth '+imgFpath+' '+str(i+1)+'/'+str(nImg)
			#read background file image
			backgroundFpath = backgroundList[i%nBackground]
			
			#read image to create mask
			mask = cv2.imread(imgFpath,0)
			#read background image file
			bgImage = cv2.imread(backgroundFpath)
			#get height, width, channel of image
			rows, cols = mask.shape
			bgRows, bgCols, bgCh = bgImage.shape

			#check the image orientation
			if rows >= cols:
				#if image = portrait then background set to portrait
				if bgRows <= bgCols:
					bgImage = np.rot90(bgImage)
			else:
				#if image = landscape then background set to landscape
				if bgRows > bgCols:
					bgImage = np.rot90(bgImage)

			#fit background size to image size
			bgImage = cv2.resize(bgImage,(cols,rows))
			####create mask to get the image area without background#####
			morphKernel = np.ones((morphKSize, morphKSize),np.uint8)

			#mask = cv2.blur(mask, (3,3))
			_,mask = cv2.threshold(mask,threshValue,255,cv2.THRESH_BINARY_INV)
			mask = cv2.erode(mask,morphKernel,1)
			mask = cv2.dilate(mask,morphKernel,1)

			randDeg = randint(0, 2)
			#masking the image
			#0 deg
			#read image file
			if randDeg == 0:
				img = cv2.imread(imgFpath)
				for y in range(0,rows):
					for x in range(0,cols):
						if mask[y,x] == 0:
							img[y,x,:] = bgImage[y,x,:]
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synthBG_0.jpg")
				cv2.imwrite(outFpath,img)
			elif randDeg == 1:	###############+ 15 deg ##########################################
				imgP15 = cv2.imread(imgFpath)
				maskP15 = np.array(mask, copy=True)

				M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
				imgP15 = cv2.warpAffine(imgP15,M,(cols,rows)) 
				
				M = cv2.getRotationMatrix2D((cols/2,rows/2),15,1)
				maskP15 = cv2.warpAffine(maskP15,M,(cols,rows)) 

				for y in range(0,rows):
					for x in range(0,cols):
						if maskP15[y,x] == 0:
							imgP15[y,x,:] = bgImage[y,x,:]
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_p15.jpg")
				cv2.imwrite(outFpath,imgP15)
				###################-15 deg########################################
				imgM15 = cv2.imread(imgFpath)
				maskM15 = np.array(mask, copy=True)
			elif randDeg == 2:
				M = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)
				imgM15 = cv2.warpAffine(imgM15,M,(cols,rows)) 
				
				M = cv2.getRotationMatrix2D((cols/2,rows/2),-15,1)
				maskM15 = cv2.warpAffine(maskM15,M,(cols,rows)) 

				for y in range(0,rows):
					for x in range(0,cols):
						if maskM15[y,x] == 0:
							imgM15[y,x,:] = bgImage[y,x,:]
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_m15.jpg")
				cv2.imwrite(outFpath,imgM15)
			
def synthNoise():
	assert len(params) == 3
	dataset = params[0]
	#set the minimum value noise value
	minVal = int(params[1])
	#set the maximum value noise value
	maxVal = int(params[2])
	pass

def synthBlur(params, inDir, outDir):
	assert len(params) == 4
	dataset = params[0]
	#determine the blur type 
	blurType = params[1]
	#set the minimum value blur kernel size value
	minKSize = int(params[2])
	#set the maximum value blur kernel size value
	maxKSize = int(params[3])
	
	outDir = os.path.join(outDir,dataset+'-synthBlur-'+blurType)
	if os.path.exists(outDir): shutil.rmtree(outDir)
	os.makedirs(outDir)

	inDir = os.path.join(inDir,dataset)
	classList = os.listdir(inDir)
	for c in classList:
		if c ==".gitignore":
			continue
		cInDir = os.path.join(inDir,c)
		imgFnames = [f for f in os.listdir(cInDir) if os.path.isfile(os.path.join(cInDir,f)) and (f.endswith(".jpg") or f.endswith('.JPG'))]
		nImg = len(imgFnames)

		cOutDir = os.path.join(outDir,c)
		if os.path.exists(cOutDir): shutil.rmtree(cOutDir)
		os.makedirs(cOutDir)

		for i,imgFname in enumerate(imgFnames):
			imgFpath = os.path.join(cInDir,imgFname)
			print 'blurring '+imgFpath+' '+str(i+1)+'/'+str(nImg)
			img = cv2.imread(imgFpath)
			#generate random kernel size
			randKSize = randint(minKSize, maxKSize)
			#make sure the kernel size is odd number, if the number is even then subtract with one
			if randKSize%2 == 0:
				randKSize -= 1
			#generate random blur type 0 = mean, 1 = median, 2 = gaussian
			if blurType == 'mean':
				mnblur = cv2.blur(img,(randKSize, randKSize))
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_mnblur_k"+str(randKSize)+".jpg")
				cv2.imwrite(outFpath,mnblur)
			elif blurType == 'median':
				mdblur = cv2.medianBlur(img, randKSize)
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_mdblur_k"+str(randKSize)+".jpg")
				cv2.imwrite(outFpath,mdblur)
			elif blurType == 'gaussian':
				gsblur = cv2.GaussianBlur(img, (randKSize, randKSize), 0)
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_gsblur_k"+str(randKSize)+".jpg")
				cv2.imwrite(outFpath,gsblur)
			elif blurType == 'random':
				randBlurType = randint(0,2)
				if randBlurType == 0:
					mnblur = cv2.blur(img,(randKSize, randKSize))
					outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_mnblur_k"+str(randKSize)+".jpg")
					cv2.imwrite(outFpath,mnblur)
				elif randBlurType == 1:
					mdblur = cv2.blur(img,(randKSize, randKSize))
					outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_mdblur_k"+str(randKSize)+".jpg")
					cv2.imwrite(outFpath,mdblur)
				elif randBlurType == 2:
					gsblur = cv2.blur(img,(randKSize, randKSize))
					outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_gsblur_k"+str(randKSize)+".jpg")
					cv2.imwrite(outFpath,gsblur)
			else:
				#if use all blur type
				mnblur = cv2.blur(img, (randKSize, randKSize))
				mdblur = cv2.medianBlur(img, randKSize)
				gsblur = cv2.GaussianBlur(img, (randKSize, randKSize), 0)
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_mnblur_k"+str(randKSize)+".jpg")
				cv2.imwrite(outFpath,mnblur)
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_mdblur_k"+str(randKSize)+".jpg")
				cv2.imwrite(outFpath,mdblur)
				outFpath = os.path.join(cOutDir,imgFname[0:-4]+"_synth_gsblur_k"+str(randKSize)+".jpg")
				cv2.imwrite(outFpath,gsblur)		

if __name__ == '__main__':
    tic = time.time()
    main(sys.argv)
    print("--- %s seconds ---" % (time.time() - tic))


