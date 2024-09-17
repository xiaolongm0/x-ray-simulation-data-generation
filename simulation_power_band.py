import numpy as np
import matplotlib.pyplot as plt

class Peak:
    def __init__(self,sigmaX,sigmaY,muX,muY,intensity):
        '''
        sigmaX and sigmaY are the standard deviations in the x and y directions
        muX and muY are the means in the x and y directions
        intensity is the intensity of the peak
        '''
        self.sigmaX = sigmaX # width in x
        self.sigmaY = sigmaY # width in y
        self.muX = muX # center in x
        self.muY = muY # center in y
        self.intensity = intensity # intensity

def powderBand(x,y,peakVals):
    '''
    x and y are the coordinates of the image
    peakVals is a Peak object
    '''
    factor = 1  #/ (2* np.pi * peakVals.sigmaX * peakVals.sigmaY)
    exponent = np.exp(-((x - peakVals.muX) ** 2 / (2 * peakVals.sigmaX ** 2) + (y - peakVals.muY) ** 2 / (2 * peakVals.sigmaY ** 2)))
    return peakVals.intensity * factor * exponent

def main():
    im = np.zeros((9424,1500))
    print(im.shape)
    rCen = 400 # center of the peak in the y direction
    nPeaks = 2 # number of peaks
    peaks = []
    etaExtent = 120 # width of the peak in the x direction
    rwidth = 20 # width of the peak in the y direction

    for peakNr in range(nPeaks):
        peaks.append(Peak(sigmaX=50000*np.random.uniform(low=0.5,high=1), sigmaY=rwidth, muX=np.random.uniform(low=0,high=9424/(peakNr+1)), muY=rCen, intensity=np.random.uniform(low=1000,high=2000)))

    y = np.linspace(0,9424,num=9424)
    x = np.linspace(rCen-etaExtent,rCen + etaExtent+1,num=etaExtent*2+1)

    Y,X = np.meshgrid(x,y)

    print("x.shape: ",x.shape)
    print("y.shape: ",y.shape)
    print("X.shape: ",X.shape)
    print("Y.shape: ",Y.shape)

    for peak in peaks:
        im[:,rCen-etaExtent:rCen+etaExtent+1] += powderBand(X,Y,peak)

    im = im.astype(np.uint16)

    plt.imshow(im.T,aspect='auto')
    plt.show()
    
# main
if __name__ == "__main__":
    main()
