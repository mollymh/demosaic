import numpy

def bilinear(cfa, pattern='GBRG', maxCount=65535):
   """
   title::
      bilinear

   description::
      This method will perform color filter array interpolation on a
      generic document mode (Bayer) image.  A simple neighbor averaging
      approach will be taken for each of the three patterned color
      channels.

      The green channel interpolation will be performed, at each "x"
      location,

         G x G x G
         x G x G x
         G x G x G
         x G x G x
         G x G x G

      as the average of the surrounding four G values.

      For the red channel

         x R x R x
         y y y y y
         x R x R x
         y y y y y
         x R x R x

      the interpolation at each "x" location will be the average of the
      horizontal neighbors.  Subsequently, the interpolation at each
      "y" location will be the average of the vertical neighbors (half 
      of which are original red values and half of which are interpolated
      red values).

      Similarly for the blue channel

         y y y y y
         B x B x B
         y y y y y
         B x B x B
         y y y y y

      the interpolation at each "x" location will be the average of the
      horizontal neighbors.  Subsequently, the interpolation at each
      "y" location will be the average of the vertical neighbors (half 
      of which are original blue values and half of which are interpolated
      blue values).

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

   # Interpolate missing green values as the average of the surrounding
   # four (4) green filtered pixels
   interpolant = (numpy.roll(g,  1, axis=1) + \
                  numpy.roll(g, -1, axis=1) + \
                  numpy.roll(g,  1, axis=0) + \
                  numpy.roll(g, -1, axis=0)) / 4

   index = numpy.where(gMask == 0)
   if len(index[0]) > 0:
      g[index] = interpolant[index]

   # Interpolate missing red values as the average of their horizontal
   # neighbors in the red/green rows, and as the average between these
   # interpolated rows
   interpolant = (numpy.roll(r,  1, axis=1) + \
                  numpy.roll(r, -1, axis=1)) / 2

   index = numpy.where(rMask == 0)
   if len(index[0]) > 0:
      r[index] = interpolant[index]

   rMask += numpy.roll(rMask,  1, axis=1)

   interpolant = (numpy.roll(r,  1, axis=0) + \
                  numpy.roll(r, -1, axis=0)) / 2

   index = numpy.where(rMask == 0)
   if len(index[0]) > 0:
      r[index] = interpolant[index]

   # Interpolate missing blue values as the average of their horizontal
   # neighbors in the blue/green rows, and as the average between these
   # interpolated rows
   interpolant = (numpy.roll(b,  1, axis=1) + \
                  numpy.roll(b, -1, axis=1)) / 2

   index = numpy.where(bMask == 0)
   if len(index[0]) > 0:
      b[index] = interpolant[index]

   bMask += numpy.roll(bMask,  1, axis=1)

   interpolant = (numpy.roll(b,  1, axis=0) + \
                  numpy.roll(b, -1, axis=0)) / 2

   index = numpy.where(bMask == 0)
   if len(index[0]) > 0:
      b[index] = interpolant[index]

   # Convert the floating-point interpolated values to match the input 
   # CFA data type
   r = r.astype(dataType)
   g = g.astype(dataType)
   b = b.astype(dataType)

   # Return the interpolated color channels as a tuple
   return r, g, b
