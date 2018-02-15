import cv2
import ipcv.demosaic
import time
import os.path

home = os.path.expanduser('~')
filename = home + os.path.sep + 'src/python/modules/ipcv/demosaic/images/demosaickTest.bayer.tif'
pattern = 'GBRG'
maxCount = 255
scale = 256

cfa = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
print('CFA ... (min, max) = ({0}, {1})'.format(cfa.min(), cfa.max()))

startTime = time.clock()
r, g, b = ipcv.demosaic.bilinear(cfa, pattern=pattern, maxCount=maxCount)
elapsedTime = time.clock() - startTime
print('Elapsed time = {0} [s]'.format(elapsedTime))
print('Interpolated R ... (min, max) = ({0}, {1})'.format(r.min(), r.max()))
print('Interpolated G ... (min, max) = ({0}, {1})'.format(g.min(), g.max()))
print('Interpolated B ... (min, max) = ({0}, {1})'.format(b.min(), b.max()))

rgb = cv2.merge((r,g,b))

cv2.imwrite('test_cfa_bilinear.tif', rgb * scale)


startTime = time.clock()
r, g, b = ipcv.demosaic.laroche_and_prescott(cfa, pattern=pattern, 
                                                  maxCount=maxCount)
elapsedTime = time.clock() - startTime
print('Elapsed time = {0} [s]'.format(elapsedTime))
print('Interpolated R ... (min, max) = ({0}, {1})'.format(r.min(), r.max()))
print('Interpolated G ... (min, max) = ({0}, {1})'.format(g.min(), g.max()))
print('Interpolated B ... (min, max) = ({0}, {1})'.format(b.min(), b.max()))

rgb = cv2.merge((r,g,b))

cv2.imwrite('test_cfa_laroche_and_prescott.tif', rgb * scale)
