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

img = np.zeros((nPxRad,nPxEta)).astype(np.uint16)

img_mask = np.zeros((nPxRad,nPxEta)).astype(np.uint16)

# define the number of rings
nRings = 30

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

rads = generate_random_rings_list(nRings, minRad, nPxRad-(minRad+100), 8)
peakPositions = [] # list to store all of the peak positions

for ringNr in range(nRings):  # number of rings
	radCen = int(rads[ringNr])

	# number of connected areas in this ring (less than maxPeaksRing)
	nSegmentsInRing = np.random.randint(0,maxSegmentsInRing)

	# Generate random segments for the ring
	segments = generate_segments(nSegmentsInRing, total_length=nPxEta, min_gap=segDistance, min_segment_length=minSegWidth, max_segment_length=maxSegWidth)

	segmend_idx = 0
	for segment in segments:
		segmend_idx += 1
		etaCen = int(segment[0] + (segment[1] - segment[0]) / 2)
		etaWidth = segment[1] - segment[0]

		numPeaks = np.random.randint(2,maxNPeaksInSegment) # Define the num of peaks in the connected area
		print(f"connect area center position: {etaCen} - connected area width: {etaWidth} - number of peaks: {numPeaks}")

		for peakNr in range(numPeaks):
			print(f"Total {numPeaks} peaks and Peak index: {peakNr} in segment {segmend_idx}")
			peakCenRad = (radCen + np.random.random(1)*rWidth).item() # y position of the peak
			peakCenEta = (etaCen + np.random.random(1)*etaWidth).item() # x position of the peak
			rWidthPeak = 1 + np.random.random(1).item()*(sigmaRMax-1)
			etaWidthPeak = 2 + np.random.random(1).item()*(sigmaEtaMax-1)
			x = np.linspace(-int(nSigmas*ceil(rWidthPeak)),int(nSigmas*ceil(rWidthPeak)),endpoint=True,num=(2*int(nSigmas*ceil(rWidthPeak))+1))
			y = np.linspace(-int(nSigmas*ceil(etaWidthPeak)),int(nSigmas*ceil(etaWidthPeak)),endpoint=True,num=(2*int(nSigmas*ceil(etaWidthPeak))+1))
			X,Y = np.meshgrid(x,y)
			Z = gaussian_2d(X,Y,sigma_x=rWidthPeak,sigma_y=etaWidthPeak,intensity=np.random.randint(maxIntensity))
			
			#plt.imshow(Z)
			#plt.imshow(Z.astype(np.uint16))
			#plt.imshow(np.log(Z.astype(np.uint16)))

			#a = np.transpose(Z).astype(np.uint16)
			#print(np.sort(a.ravel())[-200:])

			xStart = int(peakCenRad)-int(nSigmas*ceil(rWidthPeak))
			yStart = int(peakCenEta)-int(nSigmas*ceil(etaWidthPeak))
			if xStart< 0: continue
			if yStart< 0: continue
			if xStart+x.shape[0]>nPxRad: continue
			if yStart+y.shape[0]>nPxEta: continue
			peak_img = img[xStart:xStart+x.shape[0],yStart:yStart+y.shape[0]]
			peak_img += np.transpose(Z).astype(np.uint16)
			peakPositions.append([peakCenRad,peakCenEta])
			print(f"Peak center position: {peakCenEta} - {peakCenRad}")

			# plot the segment image
			#plt.imshow(peak_img)
			#plt.imshow(np.log(peak_img))
			#plt.show()

# for each ring we need to label the ring based on the connected area length

	'''
	# for each generated segment in the ring we need to label the segment based on the connected area length
	for segment_idx in range(len(segments)):
		segment = segments[segment_idx]
		segment_img = img[segment[0]:segment[1], :]
		
		# scan the segment by x axis
		connected_areas = []
		start = None
		for i in range(segment_img.shape[0]):
			if np.count_nonzero(segment_img[i, :]) > 5:
				if start is None:
					start = i
			else:
				if start is not None:
					end = i
					connected_areas.append((start, end))
					start = None
		
		# label the segment based on the connected area length
		for i, area in enumerate(connected_areas):
			area_length = area[1] - area[0]
			print(f"Segment {segment_idx+1}, Connected Area {i+1} Length: {area_length}")
	'''

peakPositions = np.array(peakPositions)
#plt.imsave('output.png',img, cmap='gray', format='png')

# max value in the img pixel values
maxValue = np.max(img)
print(f"max value: {maxValue}")

# and its location
maxValueLocation = np.where(img == maxValue)

print(f"max value location: {maxValueLocation}")

plt.imshow(np.log(img))
#plt.imsave('output1.png',img)

# plt.scatter(peakPositions[:,1],peakPositions[:,0])
plt.show()
