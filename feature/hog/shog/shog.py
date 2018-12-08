import cv2
import skimage.io as io
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.externals import joblib

def extract(imgFpath,params,outdir,tag):
    image = cv2.imread(imgFpath)
    if image is None:

      image = io.imread(imgFpath,as_grey=True)
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = gray.copy()
    
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 9  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    hog_feats = hog.compute(img)
    with open(os.path.join(outdir,tag+'_shog.pkl'),'wb') as f: joblib.dump(hog_feats,f)

def load(xrawList,yList,dirpath):
    xList = []
    for i,xraw in enumerate(xrawList):
        pklPath = os.path.join(dirpath,yList[i],xraw[0:-4]+'_shog.pkl')
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

# hog_feats = hog.compute(img)\
#                .reshape(n_cells[1] - block_size[1] + 1,
#                         n_cells[0] - block_size[0] + 1,
#                         block_size[0], block_size[1], nbins) \
#                .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
# # hog_feats now contains the gradient amplitudes for each direction,
# # for each cell of its group for each group. Indexing is by rows then columns.

# gradients = np.zeros((n_cells[0], n_cells[1], nbins))

# # count cells (border cells appear less often across overlapping groups)
# cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

# for off_y in range(block_size[0]):
#     for off_x in range(block_size[1]):
#         gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
#                   off_x:n_cells[1] - block_size[1] + off_x + 1] += \
#             hog_feats[:, :, off_y, off_x, :]
#         cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
#                    off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

# # Average gradients
# gradients /= cell_count

# # Preview
# plt.figure()
# plt.imshow(img, cmap='gray')
# plt.show()

# bin = 5  # angle is 360 / nbins * direction
# plt.pcolor(gradients[:, :, bin])
# plt.gca().invert_yaxis()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.show()
