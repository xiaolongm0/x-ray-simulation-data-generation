import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import random

# Set random seed for reproducibility
random.seed(2)
np.random.seed(2)

def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1,intensity=1):
	"""
	Compute the value of a 2D Gaussian function with means mu_x, mu_y and standard deviations sigma_x, sigma_y.
	"""
	factor = 1 / (2 * np.pi * sigma_x * sigma_y)
	exponent = np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))
	return intensity * factor * exponent

def generate_random_rings_list(num_count, lower_bound, upper_bound, min_difference):
	# Generate a list of random numbers with a minimum difference limitation between any two numbers.
    numbers = []
    while len(numbers) < num_count:
        potential_number = random.randint(lower_bound, upper_bound)
        if all(abs(potential_number - number) > min_difference for number in numbers):
            numbers.append(potential_number)
    return sorted(numbers)

def generate_segments(n, total_length=512, min_gap=5, min_segment_length=21, max_segment_length=200):
    # List to store segments, each represented as (start, end)
    segments = []

    for _ in range(n):
        # Ensure new segment does not overlap with existing ones, and adheres to length constraints
        attempt = 0  # To prevent infinite loops
        while attempt < 1000:
            attempt += 1
            start = random.randint(0, total_length - min_segment_length - min_gap) # Adjust start to consider minimum segment length and gap
            # Ensure segment length is within the defined limits
            max_end = min(start + max_segment_length, total_length)
            end = random.randint(max(start + min_segment_length, start + 1), max_end) # Ensure minimum length

            # Check if new segment overlaps with existing segments
            overlap = False
            for seg in segments:
                if not (end + min_gap <= seg[0] or start >= seg[1] + min_gap):
                    overlap = True
                    break

            if not overlap:
                # Add new segment, including gaps on both ends
                segments.append((start, end))
                break

        if attempt == 1000:
            print("Reached maximum number of attempts, may not add more segments.")
            break

    return segments

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

minSegWidth = 20
maxSegWidth = 200
segDistance = 10

maxNPeaksInSegment = 50 # maximum number of peaks in a connected area
maxSegmentsInRing = 100 # maximum number of connected areas in a ring

# rads = minRad + np.random.random(nRings)*(nPxRad-(minRad+100)) # randomize the radius of the rings (not too close to the edge) and gap larger than 8 or 10

rads = generate_random_rings_list(nRings, minRad, nPxRad-(minRad+100), 8)
#rads = generate_random_numbers(nRings, 20, 500, 8)

print(rads)

peakPositions = [] # list to store all of the peak positions

for ringNr in range(nRings):  # number of rings
	radCen = int(rads[ringNr])

	# number of connected areas in this ring (less than maxPeaksRing)
	nSegmentsInRing = np.random.randint(0,maxSegmentsInRing)
	
	# TODO: This will cause overlap of the connected areas in the same ring (fix it)
	# etaCens = minEta + np.random.random(nSegmentsInRing)*(nPxEta-(minEta+10)) # gebnerate random eta centers for the connected areas

	# Generate random segments for the ring
	segments = generate_segments(nSegmentsInRing, total_length=nPxEta, min_gap=segDistance, min_segment_length=minSegWidth, max_segment_length=maxSegWidth)

	for segment_idx in range(len(segments)):
		etaCen = int(segments[segment_idx][0] + (segments[segment_idx][1] - segments[segment_idx][0]) / 2)
		etaWidth = segments[segment_idx][1] - segments[segment_idx][0]

		numPeaks = np.random.randint(2,maxNPeaksInSegment) # Define the num of peaks in the connected area
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

			a = np.transpose(Z).astype(np.uint16)
			print(np.sort(a.ravel())[-200:])

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
#plt.imsave('output.png',img, cmap='gray', format='png')
plt.imshow(np.log(img))
plt.show()

#plt.savefig('outout.png')
# plt.scatter(peakPositions[:,1],peakPositions[:,0])
#plt.show()
