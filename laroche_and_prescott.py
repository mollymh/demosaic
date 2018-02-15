import numpy

def laroche_and_prescott(cfa, pattern='GBRG', maxCount=65535):
   """
   title::
      laroche_and_prescott

   description::
      This method will perform color filter array interpolation on a
      generic document mode (Bayer) image.  A gradient-based approach
      is taken to avoid interpolating across edges to minimize color
      artifacting.  The technique is an implementation of the following
      paper:

         C.A. Laroche and M.A. Prescott, "Apparatus and method
         for adaptively interpolating a full color image utilizing
         chrominance gradients", U.S. Patent No. 5,373,322 (1994)

   attributes::
      cfa
         A 2-dimensional ndarray containing the RAW document mode image
         to be interpolated.  The image can be any bit depth and the
         output will match the input data type.
      pattern
         A string defining the CFA layout [default is 'GBRG']:
            'GBRG'  -  G B  Raspberry Pi Camera (OmniVision OV5647)
                       R G
            'GRBG'  -  G R
                       B G
            'BGGR'  -  B G
                       G R
            'RGGB'  -  R G
                       G B
      maxCount
         The upper limit on digital count that will be permitted in
         the interpolated image.  Values that exceed this count will be
         truncated to this value. [default is 65535]

   author::
      Carl Salvaggio

   copyright::
      Copyright (C) 2015, Rochester Institute of Technology

   license::
      GPL

   version::
      1.0.0

   disclaimer::
      This source code is provided "as is" and without warranties as to 
      performance or merchantability. The author and/or distributors of 
      this source code may have made statements about this source code. 
      Any such statements do not constitute warranties and shall not be 
      relied on by the user in deciding whether to use this source code.
      
      This source code is provided without any express or implied warranties 
      whatsoever. Because of the diversity of conditions and hardware under 
      which this source code may be used, no warranty of fitness for a 
      particular purpose is offered. The user is advised to test the source 
      code thoroughly before relying on it. The user must assume the entire 
      risk of using the source code.
   """

   # Determine the dimensions of the source image
   dimensions = cfa.shape
   numberRows = dimensions[0]
   numberColumns = dimensions[1]
   if len(dimensions) == 3:
      numberBands = dimensions[2]
   else:
      numberBands = 1
   dataType = cfa.dtype

   # If the provided source image is not greyscale, raise an error
   if numberBands != 1:
      msg = 'Provide CFA image must have only 1 band: %s' % numberBands
      raise ValueError(msg)

   # Construct color filter masks for each of the individual colors using
   # the provided mask pattern
   rMask = numpy.zeros((numberRows, numberColumns), dtype=int)
   bMask = numpy.zeros((numberRows, numberColumns), dtype=int)
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
   gMask = numpy.ones((numberRows, numberColumns), dtype=int) - rMask - bMask

   # Create a floating-point array for each of the interpolated channels
   fCFA = cfa.astype(float)
   r = rMask * fCFA
   g = gMask * fCFA
   b = bMask * fCFA

   # Compute vertical (alpha) and horizontal (beta) edge classifiers
   alpha = numpy.fabs((numpy.roll(fCFA,  2, axis=1) + \
                       numpy.roll(fCFA, -2, axis=1)) / 2 - fCFA)
   beta  = numpy.fabs((numpy.roll(fCFA,  2, axis=0) + \
                       numpy.roll(fCFA, -2, axis=0)) / 2 - fCFA)

   # Interpolate missing green values based on the relative magnitudes of
   # the horizontal and vertical edge classifiers at the missing locations
   interpolantH = (numpy.roll(fCFA,  1, axis=1) + \
                   numpy.roll(fCFA, -1, axis=1)) / 2

   interpolantV = (numpy.roll(fCFA,  1, axis=0) + \
                   numpy.roll(fCFA, -1, axis=0)) / 2

   interpolantC = (interpolantH + interpolantV) / 2

                                                      # Horizontal edge
   index = numpy.where((gMask == 0) * (alpha < beta))
   if len(index[0]) > 0:
      g[index] = interpolantH[index]

                                                      # Vertical edge
   index = numpy.where((gMask == 0) * (alpha > beta))
   if len(index[0]) > 0:
      g[index] = interpolantV[index]

                                                      # Constant region
   index = numpy.where((gMask == 0) * numpy.isclose(alpha, beta, 0, 0.1))
   if len(index[0]) > 0:
      g[index] = interpolantC[index]

   # Compute the red-green and blue-green color differences at all red and blue
   # filter locations
   colorDifference = fCFA - g

   # Interpolate missing red values at all the green and blue filter locations
   # using a red-green color difference
   interpolantUR = (numpy.roll(colorDifference,  1, axis=1) + \
                    numpy.roll(colorDifference, -1, axis=1)) / 2 + g

   interpolantLL = (numpy.roll(colorDifference,  1, axis=0) + \
                    numpy.roll(colorDifference, -1, axis=0)) / 2 + g

   interpolantLR = \
      (numpy.roll(numpy.roll(colorDifference,  1, axis=1),  1, axis=0) + \
       numpy.roll(numpy.roll(colorDifference, -1, axis=1),  1, axis=0) + \
       numpy.roll(numpy.roll(colorDifference,  1, axis=1), -1, axis=0) + \
       numpy.roll(numpy.roll(colorDifference, -1, axis=1), -1, axis=0)) / 4 + g

   index = numpy.where((gMask == 1) * (numpy.roll(rMask, 1, axis=1) == 1))
   if len(index[0]) > 0:
      r[index] = interpolantUR[index]

   index = numpy.where((gMask == 1) * (numpy.roll(rMask, 1, axis=0) == 1))
   if len(index[0]) > 0:
      r[index] = interpolantLL[index]

   index = numpy.where((gMask == 0) * (bMask == 1))
   if len(index[0]) > 0:
      r[index] = interpolantLR[index]

   # Interpolate missing blue values at all the green and red filter locations
   # using a blue-green color difference
   index = numpy.where((gMask == 1) * (numpy.roll(bMask, 1, axis=1) == 1))
   if len(index[0]) > 0:
      b[index] = interpolantUR[index]

   index = numpy.where((gMask == 1) * (numpy.roll(bMask, 1, axis=0) == 1))
   if len(index[0]) > 0:
      b[index] = interpolantLL[index]

   index = numpy.where((gMask == 0) * (rMask == 1))
   if len(index[0]) > 0:
      b[index] = interpolantLR[index]

   # Trim the interpolated values to fall within acceptable count range
   index = numpy.where(r < 0)
   if len(index[0]) > 0:
      r[index] = 0
   index = numpy.where(r > maxCount)
   if len(index[0]) > 0:
      r[index] = maxCount

   index = numpy.where(g < 0)
   if len(index[0]) > 0:
      g[index] = 0
   index = numpy.where(g > maxCount)
   if len(index[0]) > 0:
      g[index] = maxCount

   index = numpy.where(b < 0)
   if len(index[0]) > 0:
      b[index] = 0
   index = numpy.where(b > maxCount)
   if len(index[0]) > 0:
      b[index] = maxCount

   # Convert the floating-point interpolated values to match the input 
   # CFA data type
   r = r.astype(dataType)
   g = g.astype(dataType)
   b = b.astype(dataType)

   # Return the interpolated color channels as a tuple
   return r, g, b
