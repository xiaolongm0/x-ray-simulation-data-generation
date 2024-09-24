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

    # Add Powerband background
    for peakNr in range(nPeaks):
        rand_width = 10000*np.random.uniform(low=0.5,high=1)
        rand_cent = np.random.randint(low=0,high=9424)
        rand_intensity = np.random.uniform(low=1000,high=2000)
        print("rand_width: ", rand_width)
        print("rand_cent: ", rand_cent)
        print("rand_intensity: ", rand_intensity)
        peaks.append(Peak(sigmaX=rand_width, sigmaY=rwidth, muX=rand_cent, muY=rCen, intensity=rand_intensity))

    y = np.linspace(0,9424,num=9424)
    x = np.linspace(rCen-etaExtent,rCen + etaExtent+1,num=etaExtent*2+1)

    Y,X = np.meshgrid(x,y)

    for peak in peaks:
        im[:,rCen-etaExtent:rCen+etaExtent+1] += powderBand(X,Y,peak)

    # Add peaks on power band
    #for i in range(100):
    #    peak = Peak(sigmaX=20*np.random.uniform(low=0.5,high=1), sigmaY=4, muX=np.random.uniform(low=0,high=9424), muY=rCen, intensity=np.random.uniform(low=3000,high=5000))
    #    im[:,rCen-etaExtent:rCen+etaExtent+1] += powderBand(X,Y,peak)

    # Plot the image
    plt.imshow(im.T,aspect='auto')
    plt.show()

    # Plot statistic
    imt = im.T
    
    # Get pixel values along the center line in both directions
    x_values = imt[400, :]  # Middle row in y direction
    y_values = imt[:, 400]   # Middle column in x direction

    # Plot the pixel values
    plt.figure(figsize=(12, 6))

    # X direction
    plt.subplot(1, 2, 1)
    # plot x axis start from 0 to 1500
    plt.ylim(0, 5000)

    plt.plot(x_values)
    plt.title('Pixel Values in X Direction')
    plt.xlabel('X Position')
    plt.ylabel('Pixel Value')

    # Y direction
    plt.subplot(1, 2, 2)
    plt.plot(y_values)
    plt.title('Pixel Values in Y Direction')
    plt.xlabel('Y Position')
    plt.ylabel('Pixel Value')

    plt.tight_layout()
    plt.show()

# main
if __name__ == "__main__":
    main()
