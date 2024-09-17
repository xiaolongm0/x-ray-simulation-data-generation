import numpy as np

class Peak:
    def __init__(self,sigmaX,sigmaY,muX,muY,intensity):
        self.sigmaX = sigmaX
        self.sigmaY = sigmaY
        self.muX = muX
        self.muY = muY
        self.intensity = intensity

def powderBand(x,y,peakVals):
    factor = 1  #/ (2* np.pi * peakVals.sigmaX * peakVals.sigmaY)
    exponent = np.exp(-((x - peakVals.muX) ** 2 / (2 * peakVals.sigmaX ** 2) + (y - peakVals.muY) ** 2 / (2 * peakVals.sigmaY ** 2)))
    return peakVals.intensity * factor * exponent

im = np.zeros((9424,1500))
print(im.shape)
rCen = 400
nPeaks = 2
peaks = []
etaExtent = 120
rwidth = 20

for peakNr in range(nPeaks):
    peaks.append(Peak(50000*np.random.uniform(low=0.5,high=1),rwidth,np.random.uniform(low=0,high=9424/(peakNr+1)),rCen,np.random.uniform(low=1000,high=2000)))
    print(peaks[-1].muX)
    print(peaks[-1].sigmaX)

y = np.linspace(0,9424,num=9424)
x = np.linspace(rCen-etaExtent,rCen + etaExtent+1,num=etaExtent*2+1)

Y,X = np.meshgrid(x,y)
# print(X.shape)
# print(Y.shape)
# print(peaks[0].muX)
# print(peaks[0].muY)
# print(peaks[0].sigmaX)
# print(peaks[0].sigmaY)
# print(peaks[0].intensity)

# px.imshow(Y,aspect='auto')

for peak in peaks:
    im[:,rCen-etaExtent:rCen+etaExtent+1] += powderBand(X,Y,peak)

im = im.astype(np.uint16)