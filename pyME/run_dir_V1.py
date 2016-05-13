from pyME.motionenergy import MotionEnergy

import struct
import numpy as np

import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pylab import savefig


def readInputHeader(fileStr):
    print fileStr

    signature = 304698591
    version = 1.0

    fStream = open(fileStr, "rb")

    f_sign = struct.unpack('i', fStream.read(4))
    f_ver = struct.unpack('f', fStream.read(4))
    f_chann = struct.unpack('b', fStream.read(1))
    f_nrX = struct.unpack('i', fStream.read(4))
    f_nrY = struct.unpack('i', fStream.read(4))
    f_nrF = struct.unpack('i', fStream.read(4))

    if f_sign[0] != signature:
        print "wrong signature"
    if f_ver[0] != version:
        print "wrong version"
    if f_chann[0] != 1:
        print "wrong number of channels"

    dims = (f_nrX[0], f_nrY[0], f_nrF[0])
    print "Stimulus dimensions: {0}x{1}x{2}".format(dims[0], dims[1], dims[2])

    return (fStream, dims)


def readInputFrame(fStream, dims):
    width, height, length = dims

    frame = np.zeros([width, height])
    for y in xrange(height):
        for x in xrange(width):
            frame[x, y] = struct.unpack('B', fStream.read(1))[0]

    return frame


fileStr = "inpGratingPlaid_gray_32x32x2400.dat"
speed = 1.5

fStream, dim = readInputHeader(fileStr)
width, height, length = dim

ME = MotionEnergy(width, height, 1)

realTimePlotting = False

nrDirs = 8
nrXY = 1024
nrFramesPerTrial = 50
nrStims = 2
nrTrials = length/nrFramesPerTrial/nrStims
print "Number of stimuli: %d" % nrStims
print "Number of trials: %d" % nrTrials
print "Number of frames per trial: %d" % nrFramesPerTrial

theta = np.arange(0, 2*np.pi, 2*np.pi/nrDirs)
theta = np.append(theta, 0)

if realTimePlotting:
    raw_input("Press Enter to continue")
    plt.ion()
    plt.figure(figsize=(15, 5), dpi=100)
    plt.show()
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(212)
    ax3 = plt.subplot(222, polar=True)

mu = np.zeros((nrStims, nrTrials, nrDirs))

for stim in xrange(nrStims):
    for trial in xrange(nrTrials):
        for i in xrange(nrFramesPerTrial):
            frame = readInputFrame(fStream, dim)

            out = ME.calcV1complex(frame, speed)
            out = out.astype(np.float32)

            d2p = (6, 5, 4, 3, 2, 1, 0, 7)
            for d in xrange(nrDirs):
                start = d * nrXY
                stop = (d + 1) * nrXY
                # add up the mean rate of each neuron, averaged over the trial
                # duration
                mu[stim][trial][d2p[d]] += np.mean(out[start:stop])

            if realTimePlotting:
                ax1.cla()
                ax1.imshow(frame.reshape(32, 32).transpose(), cmap='gray')
                ax2.cla()
                ax2.imshow(out.reshape(32, len(out)/32, order='F'),
                           cmap='gray')

                plotMu = np.append(mu[stim][trial], mu[stim][trial][0])/(i+1)
                ax3.cla()
                ax3.plot(theta, plotMu, color='r', linewidth=3)
                ax3.set_rmax(60)
                plt.draw()

    # after all the trials of a stimulus type, show tuning curve
    if not(realTimePlotting):
        if stim == 0:
            plt.figure(figsize=(15, 5), dpi=100)
            ax = dict()

        ax[stim] = plt.subplot(1, nrStims, stim + 1, polar=True)
        tt = np.arange(0, 2 * np.pi, 2 * np.pi/nrTrials)
        yy = mu[stim, :, 0]/nrTrials

        tt = np.append(tt, tt[0])
        yy = np.append(yy, yy[0])

        ax[stim].cla()
        ax[stim].plot(tt, yy, '.-r', markersize=6, linewidth=3)
        ax[stim].set_rmax(120)
        plt.draw()

        if stim == nrStims - 1:
            savefig('run_dir_V1.png')
            savefig('run_dir_V1.eps')

if realTimePlotting:
    raw_input("Press Enter to quit")

fStream.close()
