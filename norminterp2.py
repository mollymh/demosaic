import numpy as np
import ipcv
import cv2

def norminterp2(src, pattern, maxcount):
	"""
	title:
		norminterp2
	
	description:
		CFA interpolation method similar to bilinear, but
		incorporates normalized color ratios into the blue
		and red channel interpolation.
	
	attributes:
		src - single-channel document-mode image to be interpolated
		pattern - string defining CFA layout
		maxcount - max digital count in interpolated image
		
	author:
		Molly Hill, mmh5847@rit.edu
	"""

	#Creates a field of weights for edge detection
	w = np.ones(src.shape)

	WsumA = np.roll(w,1,axis=1) + \
		np.roll(w,-1,axis=1) + \
		np.roll(w,1,axis=0) + \
		np.roll(w,-1,axis=0)
	WsumB = np.roll(w,(1,1),axis=(0,1)) + \
		np.roll(w,(1,-1),axis=(0,1)) + \
		np.roll(w,(-1,1),axis=(0,1)) + \
		np.roll(w,(-1,-1),axis=(0,1))
	
	#Construct color filter masks
	rMask = np.zeros((src.shape[0], src.shape[1]), dtype=int)
	bMask = np.zeros((src.shape[0], src.shape[1]), dtype=int)
	if pattern == 'GBRG':
		rMask[1:rMask.shape[0]:2,0:rMask.shape[1]:2] = 1
		bMask[0:bMask.shape[0]:2,1:bMask.shape[1]:2] = 1
	elif pattern == 'GRBG':
		rMask[0:rMask.shape[0]:2,1:rMask.shape[1]:2] = 1
		bMask[1:bMask.shape[0]:2,0:bMask.shape[1]:2] = 1
	elif pattern == 'BGGR':
		rMask[1:rMask.shape[0]:2,1:rMask.shape[1]:2] = 1
		bMask[0:bMask.shape[0]:2,0:bMask.shape[1]:2] = 1
	elif pattern == 'RGGB':
		rMask[0:rMask.shape[0]:2,0:rMask.shape[1]:2] = 1
		bMask[1:bMask.shape[0]:2,1:bMask.shape[1]:2] = 1
	else:
		msg = 'Invalid Bayer pattern provided: %s' % pattern
		raise ValueError(msg)
	gMask = np.ones((src.shape[0], src.shape[1]), dtype=int) - rMask - bMask
	
	# Create a floating-point array for each of the interpolated channels
	fCFA = src.astype(float)
	r = rMask * fCFA
	g = gMask * fCFA
	b = bMask * fCFA
	
	gw = g*w
	
	#Interpolate Green
	Gquick = (np.roll(gw,1,axis=1) + \
		np.roll(gw,-1,axis=1) + \
		np.roll(gw,1,axis=0) + \
		np.roll(gw,-1,axis=0)) / WsumA
	index = np.where(gMask == 0)
	if len(index[0]) > 0: #what is this for?
		g[index] = Gquick[index]
	
	
	#Interpolate Red and Blue
	for i in range(2):
		rgw = r*w/g
		bgw = b*w/g
		if i == 0:
			rollX = [(1,1),(-1,1),(1,-1),(-1,1)]
			Wdiv = WsumB
		else:
			rollX = [(0,1),(0,-1),(1,0),(-1,0)]
			Wdiv = WsumA
		Rquick = g * (np.roll(rgw,rollX[0],axis=(0,1)) + \
			np.roll(rgw,rollX[1],axis=(0,1)) + \
			np.roll(rgw,rollX[2],axis=(0,1)) + \
			np.roll(rgw,rollX[3],axis=(0,1))) / Wdiv
		index = np.where(rMask == 0)
		if len(index[0]) > 0: #what is this for?
			r[index] = Rquick[index]
		rMask += np.roll(rMask,(1,1),axis=(0,1))
		
		Bquick = g * (np.roll(bgw,rollX[0],axis=(0,1)) + \
			np.roll(bgw,rollX[1],axis=(0,1)) + \
			np.roll(bgw,rollX[2],axis=(0,1)) + \
			np.roll(bgw,rollX[3],axis=(0,1))) / Wdiv
		index = np.where(bMask == 0)
		if len(index[0]) > 0: #what is this for?
			b[index] = Bquick[index]
		bMask += np.roll(bMask,(1,1),axis=(0,1))

	dst = np.stack([r,g,b],axis=2)
	
	return dst
	
if __name__ == '__main__':

	import cv2
	import os.path
	import time
	import numpy as np

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/modules/ipcv/demosaic/images/cat_trial.pgm'
	src = cv2.imread(filename,cv2.IMREAD_UNCHANGED)

	startTime = time.clock()
	dst = norminterp2(src, 'GRBG', 255)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (norminterp) = {0} [s]'.format(elapsedTime))
	
	#cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	#cv2.imshow(filename, src)

	#cv2.namedWindow(filename + ' (Filtered)', cv2.WINDOW_AUTOSIZE)
	#cv2.imshow(filename + ' (Filtered)', dst)
	
	cv2.imwrite("cattest.jpg",dst)
	
	action = ipcv.flush()

