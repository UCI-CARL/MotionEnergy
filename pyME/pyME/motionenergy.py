# import and init CUDA
import pycuda.autoinit
import pycuda.driver as cuda

import numpy as np

import Image
from pylab import show, imshow

import logging
logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.WARNING)

# motion energy device code
# SourceModule stored in variable mod
# from motion_energy_device import *
from device import *

# CUDA helper functions for alignment
# from cuda_helper import *


def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = np.int32(a)
    b = np.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)


class MotionEnergy:
    """Documentation for a class

    More details.
    """
    sizeofFloat = 4

    ##########################################################################
    # CONSTRUCTOR / DESTRUCTOR
    ##########################################################################

    def __init__(self, nrX, nrY, nrC):
        """The constructor."""

        # load params and scaling factors
        self._initParams()

        self.nrX = nrX
        self.nrY = nrY
        self.nrC = nrC

        assert nrX >= self.minNrX, "nrX must be >= %r" % self.minNrX
        assert nrY >= self.minNrY, "nrY must be >= %r" % self.minNrY

        # \TODO implement RGB support
        assert nrC == 1, "number of channels (%r) must be 1 (grayscale)" % nrC

        # initialize CUDA and establish context
        self._initCUDA()

        # initialize Motion Energy
        self._initME()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """The destructor."""
        # clean up
        self.d_resp.free()
        self.d_respV1c.free()
        self.d_stim.free()
        self.d_stimBuf.free()
        self.d_diffV1GausBufT.free()
        self.d_scalingStimBuf.free()
        self.d_v1GausBuf.free()
        self.d_diffV1GausBuf.free()
        self.d_pop.free()

        for file in self.files:
            os.unlink(file)

    ##########################################################################
    # PUBLIC METHODS
    ##########################################################################

    def calcV1complex(self, stim, speed):
        """Compute V1 complex cell responses of a frame."""

        # allocate stim on device
        self._loadInput(stim)

        # convolve the stimulus with separate V1 filters
        self._calcV1linear()

        # rectify linear response to get V1 simple cell firing rate
        self._calcV1rect()

        # spatial pooling to get V1 complex
        self._calcV1blur()

        # divisive normalization
        self._calcV1normalize()

        # steer filters in specified directions
        self._calcV1direction(speed)

        # get data from device
        res = np.zeros(self.nrX*self.nrY*self.nrDirs).astype(np.float32)
        cuda.memcpy_dtoh(res, self.d_respV1c)

        return res

    ##########################################################################
    # "PRIVATE" METHODS
    ##########################################################################

    def _accumDiffStims(self, d_resp_tmp, diffV1GausBuf, sizes, orderX,
                        orderY, orderT):
        """ Gets the responses of the filters specified in d_v1popDirs by
            interpolation.
            This is basically what shSwts.m did in the original S&H code."""

        # a useful list of factorials for computing the scaling factors for
        # the derivatives
        factorials = (1, 1, 2, 6)

        # the scaling factor for this directional derivative
        # similar to the binomial coefficients
        scale = 6/factorials[orderX]/factorials[orderY]/factorials[orderT]

        gdim = (int(iDivUp(sizes[0] * sizes[1], 256)), 1)
        bdim = (256, 1, 1)
        self.dev_accumDiffStims(
            np.intp(d_resp_tmp),
            np.intp(diffV1GausBuf),
            np.int32(sizes[0] * sizes[1]),
            np.int32(scale),
            np.int32(orderX),
            np.int32(orderY),
            np.int32(orderT),
            block=bdim, grid=gdim)

    def _calcV1linear(self):
        # compute the V1 simple cell response at different spatial scales
        # the i-th scale blurs and downsamples the image (i-1) times
        logging.debug('calcV1linear')

        sbuf = np.zeros(self.nrX * self.nrY * self.nrT).astype(np.float32)
        cuda.memcpy_dtoh(sbuf, self.d_scalingStimBuf)

        for scale in xrange(1, self.nrScales + 1):
            # blur/scale the image... each time this is called stim is
            # blurred more
            # scale==1 --> original image resolution
            # list includes self.nrScales

            if scale > 1:
                # convolve d_scalingStimBuf by scalingFilt in 3D
                d_tmp = cuda.mem_alloc(self.szXY * self.nrT)
                sizes = (self.nrX, self.nrY, self.nrT)
                self._conv3D(
                    self.d_scalingStimBuf,
                    d_tmp,
                    sizes,
                    self.d_scalingFilt,
                    np.int32(self.scalingFiltSize))

                cuda.memcpy_dtod(self.d_scalingStimBuf, d_tmp,
                                 self.szXY * self.nrT)
                d_tmp.free()  # this is a little silly, because the result
                # ends up in d_resp

            # nrT is 9, v1GaussFiltSize is 9, so we're taking
            # d_scalingStimBuf[0 ... 0+nrX*nrY*9]
            # since nrT could be greater than v1GaussFiltSize, we take "only
            # the part we want", quote Micah comment
            gdim = (int(iDivUp(self.nrX * self.nrY * self.v1GaussFiltSize,
                               256)), 1)
            bdim = (256, 1, 1)
            stimBufPt_dst = np.intp(self.d_v1GausBuf)
            offset = self.szXY * ((self.nrT - self.v1GaussFiltSize)/2)
            stimBufPt_src = np.intp(self.d_scalingStimBuf) + offset
            self.dev_memcpy_dtod(np.intp(stimBufPt_dst),
                                 np.intp(stimBufPt_src),
                                 np.int32(self.nrX * self.nrY *
                                          self.v1GaussFiltSize),
                                 block=bdim,
                                 grid=gdim)

            # convolve d_v1GausBuf by v1Gaus in 3D
            d_tmp = cuda.mem_alloc(self.szXY * self.v1GaussFiltSize)
            sizes = (self.nrX, self.nrY, self.v1GaussFiltSize)
            self._conv3D(
                self.d_v1GausBuf,
                d_tmp,
                sizes,
                self.d_v1GaussFilt,
                np.int32(self.v1GaussFiltSize))
            cuda.memcpy_dtod(self.d_v1GausBuf, d_tmp,
                             self.szXY * self.v1GaussFiltSize)

            # go through and calculate all directional derivatives and then
            # combine them to calculate the different space-time oriented
            # filters

            for orderT in xrange(0, 3 + 1):
                # reset diffV1GausBufT back to the 3D gaussian filtered
                # version
                cuda.memcpy_dtod(self.d_diffV1GausBufT, self.d_v1GausBuf,
                                 self.szXY * self.v1GaussFiltSize)

                if orderT > 0:
                    # take the derivative
                    # sizes = (self.nrX, self.nrY, self.v1GaussFiltSize)
                    self._diff(self.d_diffV1GausBufT, sizes, orderT, 2)

                for orderY in xrange(0, 3 - orderT + 1):
                    orderX = 3 - orderY - orderT

                    cuda.memcpy_dtod(self.d_diffV1GausBuf,
                                     self.d_diffV1GausBufT,
                                     self.szXY * self.v1GaussFiltSize)

                    if orderX > 0:
                        self._diff(self.d_diffV1GausBuf, sizes, orderX, 0)
                    if orderY > 0:
                        self._diff(self.d_diffV1GausBuf, sizes, orderY, 1)

                    # combine the directional derivative by the direction of
                    #  the space-time filter
                    # this is basically doing what shSwts.m did in the
                    # original S&H code
                    off1 = (scale - 1) * self.szXY * self.nrFilters
                    off2 = self.szXY * self.v1GaussFiltSize/2
                    d_respPtr = np.intp(self.d_resp) + off1
                    d_diffV1GausBufPtr = np.intp(self.d_diffV1GausBuf) + off2
                    self._accumDiffStims(d_respPtr, d_diffV1GausBufPtr, sizes,
                                         orderX, orderY, orderT)
        # \NOTE the scaling factor scaleV1linear will be applied in
        # calcV1rect()

        # consider edge effects
        # suppress filter responses at pixel locations close to image border
        length = self.nrX * self.nrY * self.nrFilters * self.nrScales
        gdim = (int(iDivUp(length, 256)), 1)
        bdim = (256, 1, 1)
        self.dev_edges(
            self.d_resp,
            np.int32(length),
            np.int32(self.nrX),
            np.int32(self.nrY),
            block=bdim, grid=gdim)

    def _calcV1rect(self):
        """Performs full-wave rectification of the linear responses."""
        logging.debug('calcV1rect')

        length = self.nrX*self.nrY*self.nrFilters*self.nrScales
        gdim = (int(iDivUp(length, 256)), 1)
        bdim = (256, 1, 1)

        self.dev_fullRect2(
            self.d_resp,
            np.int32(length),
            np.double(self.scaleV1Linear),
            np.double(self.scaleV1FullWaveRect),
            block=bdim, grid=gdim)

    def _calcV1blur(self):
        logging.debug('calcV1blur')

        d_tmp = cuda.mem_alloc(self.szXY*self.nrFilters*self.nrScales)
        sizes = (self.nrX, self.nrY, self.nrFilters*self.nrScales)
        self._conv2D(self.d_resp, d_tmp, sizes, self.d_complexV1Filt,
                     self.complexV1FiltSize)
        d_tmp.free()  # result ends up in d_resp

        length = self.nrX * self.nrY * self.nrFilters * self.nrScales
        gdim = (int(iDivUp(length, 256)), 1)
        bdim = (256, 1, 1)
        self.dev_scale(
            self.d_resp,
            np.double(self.scaleV1Blur),
            np.int32(length),
            block=bdim, grid=gdim)

    def _calcV1normalize(self):
        logging.debug('calcV1normalize')

        # we need to associate each filter at pixel position (x,y) with a
        # power/intensity, but there are 28 filter responses at each location
        # so we need to (i) average over the 28 filters (3rd dimension in
        # d_resp) and put it in d_pop
        gdim = (int(iDivUp(self.nrX*self.nrY, 128)), self.nrScales)
        bdim = (128, 1, 1)
        self.dev_mean3(
            self.d_resp,
            self.d_pop,
            np.int32(self.nrX*self.nrY),
            np.int32(self.nrFilters),
            block=bdim, grid=gdim)

        # ... (ii) scale with scaleV1Complex ...
        length = self.nrX*self.nrY*self.nrFilters*self.nrScales
        gdim = (int(iDivUp(length, 128)), 1)
        bdim = (128, 1, 1)
        self.dev_scale(
            self.d_resp,
            np.double(self.scaleV1Complex),
            np.int32(length),
            block=bdim, grid=gdim)

        # ... and (iii) sum over some spatial neighborhood for the
        # normalization
        sizes = (self.nrX, self.nrY, self.nrScales)
        d_tmp = cuda.mem_alloc(self.szXY*self.nrScales)
        self._conv2D(
            self.d_pop,
            d_tmp,
            sizes,
            self.d_normV1filt,
            self.normV1filtSize)
        d_tmp.free()  # result ends up in d_pop

        # scale with V1NormStrength and V1NormPopK
        gdim = (int(iDivUp(self.nrX * self.nrY * self.nrScales, 128)), 1)
        bdim = (128, 1, 1)
        self.dev_scale(
            self.d_pop,
            np.double(self.scaleV1NormStrength*self.scaleV1NormPopK),
            np.int32(self.nrX*self.nrY*self.nrScales),
            block=bdim, grid=gdim)

        # divisive normalization
        # d_resp is the numerator, d_pop the denominator sum term
        gdim = (int(iDivUp(self.nrX * self.nrY, 128)), self.nrScales)
        bdim = (128, 1, 1)
        self.dev_normalize(
            self.d_resp,
            self.d_pop,
            np.int32(self.nrX*self.nrY),
            np.double(self.scaleV1C50),
            block=bdim, grid=gdim)

    def _calcV1direction(self, speed):
        """Generate direction selectivity via filter interpolation.
            The 28 filter responses do now need to be collapsed onto the
            directions and speeds of motion specified in the motion
            projections.
        """
        logging.debug('calcV1direction')

        length = self.nrX*self.nrY*self.nrFilters*self.nrScales*self.nrDirs
        gdim = (int(iDivUp(length, 256)), 1)
        bdim = (256, 1, 1)
        self.dev_filt2dir(
            self.d_respV1c,
            self.d_resp,
            np.int32(length),
            np.int32(self.nrX * self.nrY),
            np.int32(self.nrScales),
            np.double(speed),
            block=bdim, grid=gdim)

        # half-wave rectification to avoid negative firing rates
        # 0 Hz spontaneous firing
        # \TODO justify scaling factors
        # print "dont't halfrect again"
        gdim = (int(iDivUp(self.nrX * self.nrY * self.nrDirs, 128)), 1)
        bdim = (128, 1, 1)
        self.dev_scaleHalfRect(
            self.d_respV1c,
            np.int32(self.nrX * self.nrY * self.nrDirs),
            np.double(self.scaleV1ComplexFiring),
            np.double(0),
            block=bdim, grid=gdim)

    def _conv2D(self, d_idata, d_odata, sizes, d_filt, filtlen):
        logging.debug("conv2D")

        # convolve the first dimension
        gdim = (int(iDivUp(sizes[0], self.CONV1_THREAD_SIZE-(filtlen-1))),
                sizes[1]*sizes[2])
        bdim = (self.CONV1_THREAD_SIZE, 1, 1)
        self.dev_conv1(
            d_idata,
            d_odata,
            np.int32(sizes[0]),
            np.intp(d_filt),
            np.int32(filtlen),
            block=bdim, grid=gdim)

        szBytes = self.sizeofFloat*reduce(lambda x, y: x*y, sizes)
        d_tmp = cuda.mem_alloc(szBytes)
        cuda.memcpy_dtod(d_tmp, d_idata, szBytes)
        cuda.memcpy_dtod(d_idata, d_odata, szBytes)
        cuda.memcpy_dtod(d_odata, d_tmp, szBytes)

        # convolve the second dimension
        gdim = (int(iDivUp(sizes[0], self.CONVN_THREAD_SIZE1)),
                int(iDivUp(sizes[1],
                    self.CONVN_THREAD_SIZE2-(filtlen - 1)) * sizes[2]))
        bdim = (self.CONVN_THREAD_SIZE1, self.CONVN_THREAD_SIZE2, 1)
        self.dev_convn(
            d_idata,
            d_odata,
            np.int32(sizes[0]),
            np.int32(sizes[1]),
            np.int32(sizes[0]),
            np.int32(sizes[0]*sizes[1]),
            np.int32(sizes[2]),
            np.intp(d_filt),
            np.int32(filtlen),
            block=bdim, grid=gdim)

    def _conv3D(self, d_idata, d_odata, sizes, d_filt, filtlen):
        logging.debug('conv3D')

        # convolve the first dimension
        gdim = (int(iDivUp(sizes[0], self.CONV1_THREAD_SIZE-(filtlen - 1))),
                sizes[1]*sizes[2])
        bdim = (self.CONV1_THREAD_SIZE, 1, 1)
        self.dev_conv1(
            d_idata,
            d_odata,
            np.int32(sizes[0]),
            np.intp(d_filt),
            np.int32(filtlen),
            block=bdim, grid=gdim)

        szBytes = self.sizeofFloat*reduce(lambda x, y: x*y, sizes)
        d_tmp = cuda.mem_alloc(szBytes)
        cuda.memcpy_dtod(d_tmp, d_idata, szBytes)
        cuda.memcpy_dtod(d_idata, d_odata, szBytes)
        cuda.memcpy_dtod(d_odata, d_tmp, szBytes)

        # convolve the second dimension
        gdim = (int(iDivUp(sizes[0], self.CONVN_THREAD_SIZE1)),
                int(iDivUp(sizes[1],
                    self.CONVN_THREAD_SIZE2 - (filtlen - 1))*sizes[2]))
        bdim = (self.CONVN_THREAD_SIZE1, self.CONVN_THREAD_SIZE2, 1)
        self.dev_convn(
            d_idata,
            d_odata,
            np.int32(sizes[0]),
            np.int32(sizes[1]),
            np.int32(sizes[0]),
            np.int32(sizes[0]*sizes[1]),
            np.int32(sizes[2]),
            np.intp(d_filt),
            np.int32(filtlen),
            block=bdim, grid=gdim)

        cuda.memcpy_dtod(d_tmp, d_idata, szBytes)
        cuda.memcpy_dtod(d_idata, d_odata, szBytes)
        cuda.memcpy_dtod(d_odata, d_tmp, szBytes)

        # convolve the third dimension
        gdim = (int(iDivUp(sizes[0], self.CONVN_THREAD_SIZE1)),
                int(iDivUp(sizes[2],
                    self.CONVN_THREAD_SIZE2 - (filtlen - 1))*sizes[1]))
        bdim = (self.CONVN_THREAD_SIZE1, self.CONVN_THREAD_SIZE2, 1)
        self.dev_convn(
            d_idata,
            d_odata,
            np.int32(sizes[0]),
            np.int32(sizes[2]),
            np.int32(sizes[0]*sizes[1]),
            np.int32(sizes[0]),
            np.int32(sizes[1]),
            np.intp(d_filt),
            np.int32(filtlen),
            block=bdim, grid=gdim)

        cuda.memcpy_dtod(d_tmp, d_idata, szBytes)
        cuda.memcpy_dtod(d_idata, d_odata, szBytes)
        cuda.memcpy_dtod(d_odata, d_tmp, szBytes)

    def _diff(self, d_iodata, sizes, order, dim):
        """Takes the derivative of iodata, returns as iodata."""

        if order == 1:
            filtlen = self.diff1filtSize
            filt = self.d_diff1filt
        elif order == 2:
            filtlen = self.diff2filtSize
            filt = self.d_diff2filt
        elif order == 3:
            filtlen = self.diff3filtSize
            filt = self.d_diff3filt
        else:
            raise NameError("Order must be in the range [1,3]")

        szBytes = self.sizeofFloat*reduce(lambda x, y: x*y, sizes)
        d_tmp_odata = cuda.mem_alloc(szBytes)

        if dim == 0:
            # convolve the first dimension
            gdim = (int(iDivUp(sizes[0],
                        self.CONV1_THREAD_SIZE - (filtlen-1))),
                    sizes[1] * sizes[2])
            bdim = (self.CONV1_THREAD_SIZE, 1, 1)
            self.dev_conv1(
                d_iodata,
                d_tmp_odata,
                np.int32(sizes[0]),
                np.intp(filt),
                np.int32(filtlen),
                block=bdim, grid=gdim)

        elif dim == 1:
            # convolve the second dimension
            gdim = (int(iDivUp(sizes[0], self.CONVN_THREAD_SIZE1)),
                    int(iDivUp(sizes[1],
                        self.CONVN_THREAD_SIZE2 - (filtlen - 1)) * sizes[2]))
            bdim = (self.CONVN_THREAD_SIZE1, self.CONVN_THREAD_SIZE2, 1)
            self.dev_convn(
                d_iodata,
                d_tmp_odata,
                np.int32(sizes[0]),
                np.int32(sizes[1]),
                np.int32(sizes[0]),
                np.int32(sizes[0]*sizes[1]),
                np.int32(sizes[2]),
                np.intp(filt),
                np.int32(filtlen),
                block=bdim, grid=gdim)

        elif dim == 2:
            # convolve the third dimension
            gdim = (int(iDivUp(sizes[0], self.CONVN_THREAD_SIZE1)),
                    int(iDivUp(sizes[2],
                        self.CONVN_THREAD_SIZE2 - (filtlen - 1)) * sizes[1]))
            bdim = (self.CONVN_THREAD_SIZE1, self.CONVN_THREAD_SIZE2, 1)
            self.dev_convn(
                d_iodata,
                d_tmp_odata,
                np.int32(sizes[0]),
                np.int32(sizes[2]),
                np.int32(sizes[0]*sizes[1]),
                np.int32(sizes[0]),
                np.int32(sizes[1]),
                np.intp(filt),
                np.int32(filtlen),
                block=bdim, grid=gdim)

        cuda.memcpy_dtod(d_iodata, d_tmp_odata, szBytes)
        d_tmp_odata.free()

    def _initCUDA(self):
        """Initializes CUDA and establishes context using pycuda.autoinit"""
        self.context = None
        self.device = pycuda.autoinit.device
        self.computecc = self.device.compute_capability()

    def _initME(self):
        """Initializes the MotionEnergy CUDA functions."""
        logging.debug('initME')

        # register all device functions for easy access
        # imported from motion_energy_device.py
        self.dev_conv1 = mod.get_function("dev_conv1")
        self.dev_convn = mod.get_function("dev_convn")
        self.dev_accumDiffStims = mod.get_function("dev_accumDiffStims")
        self.dev_filt2dir = mod.get_function("dev_filt2dir")
        self.dev_edges = mod.get_function("dev_edges")
        self.dev_fullRect2 = mod.get_function("dev_fullRect2")
        self.dev_mean3 = mod.get_function("dev_mean3")
        self.dev_normalize = mod.get_function("dev_normalize")
        self.dev_split_gray = mod.get_function("dev_split_gray")
        self.dev_split_RGB = mod.get_function("dev_split_RGB")
        self.dev_sub = mod.get_function("dev_sub")
        self.dev_ave = mod.get_function("dev_ave")
        self.dev_sum = mod.get_function("dev_sum")
        self.dev_scaleHalfRect = mod.get_function("dev_scaleHalfRect")
        self.dev_scale = mod.get_function("dev_scale")
        self.dev_split_gray = mod.get_function("dev_split_gray")
        self.dev_split_RGB = mod.get_function("dev_split_RGB")
        self.dev_memcpy_dtod = mod.get_function("dev_memcpy_dtod")

        # for quick access: the size in bytes of nrX*nrY floats
        self.szXY = self.sizeofFloat * self.nrX * self.nrY

        # V1 filter responses
        self.d_resp = cuda.mem_alloc(self.szXY*self.nrFilters*self.nrScales)

        # V1 complex cell responses
        self.d_respV1c = cuda.mem_alloc(self.szXY*self.nrDirs)

        # stim frame
        self.d_stim = cuda.mem_alloc(self.szXY*self.nrC)

        # stim frame buffer (last nrT frames)
        self.d_stimBuf = cuda.mem_alloc(self.szXY*self.nrT)
        # I'm not sure if this memset works as expected... for now, memcpy an
        # array of zeros
        # cuda.memset_d32(self.d_stimBuf, 0, self.nrX*self.nrY*self.nrT)
        tmp = np.zeros(self.nrX*self.nrY*self.nrT).astype(np.float32)
        cuda.memcpy_htod(self.d_stimBuf, tmp)

        self.d_diffV1GausBufT = cuda.mem_alloc(self.szXY*self.v1GaussFiltSize)

        self.d_scalingStimBuf = cuda.mem_alloc(self.szXY*self.nrT)
        self.d_v1GausBuf = cuda.mem_alloc(self.szXY*self.v1GaussFiltSize)
        self.d_diffV1GausBuf = cuda.mem_alloc(self.szXY*self.v1GaussFiltSize)
        self.d_pop = cuda.mem_alloc(self.szXY*self.nrScales)

        self.d_scalingFilt = mod.get_global("d_scalingFilt")[0]
        self.d_v1GaussFilt = mod.get_global("d_v1GaussFilt")[0]
        self.d_complexV1Filt = mod.get_global("d_complexV1Filt")[0]
        self.d_normV1filt = mod.get_global("d_normV1filt")[0]
        self.d_diff1filt = mod.get_global("d_diff1filt")[0]
        self.d_diff2filt = mod.get_global("d_diff2filt")[0]
        self.d_diff3filt = mod.get_global("d_diff3filt")[0]

    def _initParams(self):
        """Initializes all class attributes to default values, akin to
            shPars.m
        """
        logging.debug('initParams')

        self.nrDirs = 8
        self.nrFilters = 28
        self.nrScales = 3  # number of scales at which to filter
        self.nrT = 9
        self.v1GaussFiltSize = 9

        # from shGetDims
        # dimensions must be greater or equal these
        # \TODO compute instead of hardcoding
        self.minNrX = 19
        self.minNrY = 19

        self.scalingFiltSize = 5
        self.CONV1_THREAD_SIZE = 256
        self.CONVN_THREAD_SIZE1 = 16
        self.CONVN_THREAD_SIZE2 = 31  # 31 is faster than 32

        # S&H scaling factors
        self.scaleV1Linear = 6.6084
        self.scaleV1FullWaveRect = 1.9263
        self.scaleV1Blur = 1.0205
        self.scaleV1NormPopK = 1.0  # 0.2401
        self.scaleV1NormStrength = 0.98
        self.scaleV1Complex = 0.99
        self.scaleV1C50 = 0.1
        self.scaleV1ComplexFiring = 10.0

        # some more #define
        self.diff1filtSize = 3
        self.diff2filtSize = 3
        self.diff3filtSize = 5
        self.complexV1FiltSize = 11
        self.normV1filtSize = 25

    def _loadInput(self, stim):
        logging.debug('loadInput')

        # shortcuts
        nrXY = self.nrX * self.nrY
        nrXYD = self.nrX * self.nrY * self.nrDirs

        # parse input
        assert type(stim).__module__ == "numpy", "stim must be numpy array"
        assert type(stim).__name__ == "ndarray", "stim must be numpy.ndarray"
        assert stim.size > 0, "stim cannot be []"
        stim = stim.astype(np.ubyte)

        rows, cols = stim.shape
        logging.debug("- stim shape={0}x{1}".format(rows, cols))

        # shift d_stimBuf in time by 1 frame, from frame i to frame i-1
        # write our own memcpy kernel... :-(
        gdim = (int(iDivUp(nrXY, 128)), 1)
        bdim = (128, 1, 1)
        for i in xrange(1, self.nrT):
            stimBufPt_dst = np.intp(self.d_stimBuf) + self.szXY * (i - 1)
            stimBufPt_src = np.intp(self.d_stimBuf) + self.szXY * i
            self.dev_memcpy_dtod(
                stimBufPt_dst,
                stimBufPt_src,
                np.int32(nrXY),
                block=bdim, grid=gdim)

        # index into d_stimBuf array to place the new stim at the end
        # (newest frame at pos: nrT-1)
        d_stimBufPt = np.intp(self.d_stimBuf) + self.szXY * (self.nrT-1)

        # \TODO implement RGB support
        self.dev_split_gray(
            d_stimBufPt,
            cuda.In(stim),
            np.int32(stim.size),
            block=bdim, grid=gdim)

        # create working copy of d_stimBuf
        cuda.memcpy_dtod(self.d_scalingStimBuf, self.d_stimBuf,
                         self.szXY*self.nrT)

        # reset V1complex responses to 0
        # \FIXME not sure how to use memset...doesn't seem to give expected
        # result
        tmp = np.zeros(nrXYD).astype(np.float32)
        cuda.memcpy_htod(self.d_respV1c, tmp)

        # allocate d_resp, which will contain the response to all 28
        # (nrFilters) space-time orientations at 3 (nrScales) scales for
        # every pixel location (nrX*nrY)
        tmp = np.zeros(nrXY*self.nrFilters*self.nrScales).astype(np.float32)
        cuda.memcpy_htod(self.d_resp, tmp)
