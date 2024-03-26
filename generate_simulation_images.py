import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import random

random.seed(2)
np.random.seed(2)

def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1,intensity=1):
	"""
	Compute the value of a 2D Gaussian function with means mu_x, mu_y and standard deviations sigma_x, sigma_y.
	"""
	factor = 1 / (2 * np.pi * sigma_x * sigma_y)
	exponent = np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))
	return intensity * factor * exponent

def generate_random_numbers(num_count, lower_bound, upper_bound, min_difference):
	# Generate a list of random numbers with a minimum difference limitation between any two numbers.
    numbers = []
    while len(numbers) < num_count:
        potential_number = random.randint(lower_bound, upper_bound)
        if all(abs(potential_number - number) > min_difference for number in numbers):
            numbers.append(potential_number)
    return sorted(numbers)

# define the size of the image
nPxRad = 1500
nPxEta = 9424

#nPxRad = 512
#nPxEta = 512

img = np.zeros((nPxRad,nPxEta)).astype(np.uint16)

# define the number of rings
nRings = 30
#nRings = 10

# define the gaussian parameters
sigmaEtaMax = 4
sigmaRMax = 2
nSigmas = 20

maxIntensity = 10000 # maximum intensity of the peaks
minRad = 200
minEta = 10


rWidth = 0 # How wide should the peak centers building this peak be?
maxEtaWidth = 200

maxNPeaksInConnectedArea = 50 # maximum number of peaks in a connected area
maxConnectedAreasInRing = 100 # maximum number of connected areas in a ring

# rads = minRad + np.random.random(nRings)*(nPxRad-(minRad+100)) # randomize the radius of the rings (not too close to the edge) and gap larger than 8 or 10

rads = generate_random_numbers(nRings, minRad, nPxRad-(minRad+100), 8)
#rads = generate_random_numbers(nRings, 20, 500, 8)

print(rads)

peakPositions = [] # list to store all of the peak positions

for ringNr in range(nRings):  # number of rings
	radCen = int(rads[ringNr])

	# number of connected areas in this ring (less than maxPeaksRing)
	nConnectedAreasInRing = np.random.randint(0,maxConnectedAreasInRing)
	
	# TODO: This will cause overlap of the connected areas in the same ring (fix it)
	etaCens = minEta + np.random.random(nConnectedAreasInRing)*(nPxEta-(minEta+10)) # gebnerate random eta centers for the connected areas

	for singleConnecatedArea in range(nConnectedAreasInRing):
		etaCen = int(etaCens[singleConnecatedArea]) # connected area center postion
		# Add stretching based on radCen: for radCen=nPxRad, stretching should be 0, otherwise, we would have a factor for multiplying with etaWidth
		etaWidth = np.random.randint(20,maxEtaWidth) # Define the width of the connected area
		numPeaks = np.random.randint(2,maxNPeaksInConnectedArea) # Define the num of peaks in the connected area
		print(f"connect area center position: {etaCen} - connected area width: {etaWidth} - number of peaks: {numPeaks}")
		
		for peakNr in range(numPeaks):
			peakCenRad = (radCen + np.random.random(1)*rWidth).item()
			peakCenEta = (etaCen + np.random.random(1)*etaWidth).item()
			rWidthPeak = 1 + np.random.random(1).item()*(sigmaRMax-1)
			etaWidthPeak = 2 + np.random.random(1).item()*(sigmaEtaMax-1)
			x = np.linspace(-int(nSigmas*ceil(rWidthPeak)),int(nSigmas*ceil(rWidthPeak)),endpoint=True,num=(2*int(nSigmas*ceil(rWidthPeak))+1))
			y = np.linspace(-int(nSigmas*ceil(etaWidthPeak)),int(nSigmas*ceil(etaWidthPeak)),endpoint=True,num=(2*int(nSigmas*ceil(etaWidthPeak))+1))
			X,Y = np.meshgrid(x,y)
			Z = gaussian_2d(X,Y,sigma_x=rWidthPeak,sigma_y=etaWidthPeak,intensity=np.random.randint(maxIntensity))

			#a = np.transpose(Z).astype(np.uint16)
			#print(np.sort(a.ravel())[-200:])

			xStart = int(peakCenRad)-int(nSigmas*ceil(rWidthPeak))
			yStart = int(peakCenEta)-int(nSigmas*ceil(etaWidthPeak))
			if xStart< 0: continue
			if yStart< 0: continue
			if xStart+x.shape[0]>nPxRad: continue
			if yStart+y.shape[0]>nPxEta: continue
			# print(f"peak start position: ({xStart},{yStart}), width: {x.shape[0]} height:{y.shape[0]}")
			img[xStart:xStart+x.shape[0],yStart:yStart+y.shape[0]] += np.transpose(Z).astype(np.uint16)
			peakPositions.append([peakCenRad,peakCenEta])
			
peakPositions = np.array(peakPositions)
plt.imshow(np.log(img))
# plt.savefig('outout.png')
# plt.scatter(peakPositions[:,1],peakPositions[:,0])
plt.show()
