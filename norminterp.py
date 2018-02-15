import numpy as np
import ipcv

def norminterp(src, pattern, edge, maxcount):

	#initialize gradients

	dst = np.zeros(src.shape)
	dst = np.stack([dst,dst,dst],axis=2)
	dst.flat[1::2] = src.flat[1::2] #green
	
	padsrc = np.pad(src,(2,2),'reflect')
	w = np.ones(padsrc.shape)

	for r in range(2,src.shape[0]+2): #make this shit more efficient? with where function
		for c in range(2,src.shape[1]+2):
			Gquick = 0
			Wquick = 0
			if (r%2 == 1 and c%2 == 0) or (r%2 == 0 and c%2 == 1):
				Wquick = w[r-1][c] + w[r][c-1] \
					+ w[r+1][c] + w[r][c+1]
				Gquick = padsrc[r-1][c]*w[r-1][c]/Wquick + \
					 padsrc[r][c-1]*w[r][c-1]/Wquick + \
					 padsrc[r+1][c]*w[r+1][c]/Wquick + \
					 padsrc[r][c+1]*w[r][c+1]/Wquick
				dst[r-2][c-2][1] = Gquick

	#red and blue
	for r in range(2,src.shape[0]+1): #make this shit more efficient? with where function
		for c in range(2,src.shape[1]+1):
			Rquick = 0
			Bquick = 0
			Wquick = 0
			if r%2 == 0 and c%2 == 0: #red
				Wquick = w[r-1][c-1] + w[r-1][c+1] \
					+ w[r+1][c-1] + w[r+1][c+1]
				Xg = dst[r-2][c-2][1]
				Rquick = padsrc[r-1][c-1]*(w[r-1][c-1]/Wquick)/dst[r-3][c-3][1] + \
					 padsrc[r-1][c+1]*(w[r-1][c+1]/Wquick)/dst[r-3][c-1][1] + \
					 padsrc[r+1][c-1]*(w[r+1][c-1]/Wquick)/dst[r-1][c-3][1] + \
					 padsrc[r+1][c+1]*(w[r+1][c+1]/Wquick)/dst[r-1][c-1][1]
				Rquick = Xg * Rquick
				dst[r-2][c-2][0] = Rquick
			if r%2 == 1 and c%2 == 1: #blue
				Wquick = w[r-1][c-1] + w[r-1][c+1] \
					+ w[r+1][c-1] + w[r+1][c+1]
				Xg = dst[r-2][c-2][1]
				Bquick = padsrc[r-1][c-1]*(w[r-1][c-1]/Wquick)/dst[r-3][c-3][1] + \
					 padsrc[r-1][c+1]*(w[r-1][c+1]/Wquick)/dst[r-3][c-1][1] + \
					 padsrc[r+1][c-1]*(w[r+1][c-1]/Wquick)/dst[r-1][c-3][1] + \
					 padsrc[r+1][c+1]*(w[r+1][c+1]/Wquick)/dst[r-1][c-1][1]
				Bquick = Xg * Bquick
				dst[r-2][c-2][2] = Bquick			
	
	dst = dst.astype(np.uint8)
	#normalization later
	
	#chop off padding and make ints
	
	#ask what pattern it is
	#BGR or RGB?
	
	x = np.asarray([[1,2,3],[6,8,10],[82,39,21]])
	#print(np.roll(x,1,axis=0))
	#print(x.flat[1::2])
	
	return dst
	
if __name__ == '__main__':

	import cv2
	import os.path
	import time
	import numpy as np

	home = os.path.expanduser('~')
	filename = home + os.path.sep + 'src/python/modules/ipcv/demosaic/images/cat_trial.dng'
	
	fd = open(filename, 'rb')
	rows = 1000
	cols = 2000
	f = np.fromfile(fd, dtype=np.uint8,count=rows*cols)
	src = f.reshape((rows, cols)) #notice row, column format
	fd.close()

	startTime = time.clock()
	dst = norminterp(src, 'RGGB', 0, 255)
	elapsedTime = time.clock() - startTime
	print('Elapsed time (norminterp) = {0} [s]'.format(elapsedTime))
	
	#cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE)
	#cv2.imshow(filename, src)

	cv2.namedWindow(filename + ' (Filtered)', cv2.WINDOW_AUTOSIZE)
	cv2.imshow(filename + ' (Filtered)', dst)
	
	action = ipcv.flush()

