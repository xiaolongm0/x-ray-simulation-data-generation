import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import random
import argparse

# Set random seed for reproducibility
random.seed(2)
np.random.seed(2)
CUTOFF_PIXELS_THRESHOLD = 5

def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=1, sigma_y=1, intensity=1):
    """
	Compute the value of a 2D Gaussian function with means mu_x, mu_y and standard deviations sigma_x, sigma_y.
	"""
    factor = 1 / (2 * np.pi * sigma_x * sigma_y)
    exponent = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)))
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
            start = random.randint(0,
                                   total_length - min_segment_length - min_gap)  # Adjust start to consider minimum segment length and gap
            # Ensure segment length is within the defined limits
            max_end = min(start + max_segment_length, total_length)
            end = random.randint(max(start + min_segment_length, start + 1), max_end)  # Ensure minimum length

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
            # print("Reached maximum number of attempts, may not add more segments.")
            break
    return segments

def find_and_modify_continuous_regions(ring_img, peaks, instance_idx):
    n_cols = ring_img.shape[1]
    raw_continuous_regions = []
    peaks_arr = np.sort(np.array(peaks))

    contour_start = n_cols  # the inclusive index
    contour_end = -1  # the exclusive index
    for col in range(n_cols):
        height = np.sum(ring_img[:, col] > 0)
        if height > 0:
            if contour_start > col:
                contour_start = col
        elif contour_end < col:
            contour_end = col

        if contour_end > contour_start:
            raw_continuous_regions.append((contour_start, contour_end))
            contour_start = n_cols
            contour_end = -1

    adjusted_continuous_regions = []
    for contour in raw_continuous_regions:
        start = contour[0]
        end = contour[1]
        inner_peaks = peaks_arr[(start <= peaks_arr) & (peaks_arr < end)]
        if len(inner_peaks) > 1:
            for peak_idx in range(0, len(inner_peaks) - 1):
                peak_start = int(np.ceil(inner_peaks[peak_idx]))
                peak_end = int(np.floor(inner_peaks[peak_idx + 1]))
                if peak_end - peak_start < 1:
                    continue
                thinnest_col = peak_start
                for col in range(peak_start, peak_end + 1):
                    if np.sum(ring_img[:, col]) < np.sum(ring_img[:, thinnest_col]):
                        thinnest_col = col
                if np.sum(ring_img[:, thinnest_col]) < CUTOFF_PIXELS_THRESHOLD:
                    adjusted_continuous_regions.append((start, thinnest_col + 1))
                    start = thinnest_col + 1

        if end > start:
            adjusted_continuous_regions.append((start, end))

    for contour in adjusted_continuous_regions:
        start = contour[0]
        end = contour[1]
        #new_value = (end - start) // 8
        instance_idx += 1
        
        if instance_idx >= 255:
            new_value = 0 # once the instance label is larger than 255, reset the instance label to 0
        new_value = instance_idx
        
        #print(f"Instance {instance_idx} - new_value: {new_value}")
        
        for col in range(start, end):
            ring_img[:, col][ring_img[:, col] > 0] = new_value

    return adjusted_continuous_regions, ring_img, instance_idx

def find_continuous_regions(ring_img):
    n_cols = ring_img.shape[1]
    continuous_regions = []
    in_region = False
    start_idx = None

    for col_idx in range(n_cols):
        count = np.sum(ring_img[:, col_idx] > 0)

        if count >= 1:
            if not in_region:
                in_region = True
                start_idx = col_idx
        else:
            if in_region:
                continuous_regions.append((start_idx, col_idx - 1))
                in_region = False

    if in_region:
        continuous_regions.append((start_idx, n_cols - 1))

    return continuous_regions


def generate_simulation_image(image_idx):
    # define the size of the image
    nPxRad = 1024
    nPxEta = 1024

    img = np.zeros((nPxRad, nPxEta)).astype(np.uint16)
    mask = np.zeros((nPxRad, nPxEta)).astype(np.uint16)

    # define the number of rings
    nRings = 15

    # define the gaussian parameters
    sigmaEtaMax = 10000
    sigmaRMax = 2
    nSigmas = 20

    maxIntensity = 10000  # maximum intensity of the peaks

    minRad = 20  # distance to the upper and lower border of the image
    minEta = 10  # distance to the left and right border of the image

    rWidth = 0  # How wide should the peak centers building this peak be?

    minSegWidth = 20
    maxSegWidth = 200
    segDistance = 10

    maxNPeaksInSegment = 50  # maximum number of peaks in a connected area
    maxSegmentsInRing = 100  # maximum number of connected areas in a ring

    rads = generate_random_rings_list(nRings, minRad, nPxRad - (minRad + 100), 8)
    peakPositions = []  # list to store all of the peak positions

    instance_idx = 0 # use for record the instance index (instance means a single small segment in this image)

    for ringNr in range(nRings):  # number of rings
        tmp_mask = np.zeros((nPxRad, nPxEta)).astype(np.uint16)
        tmp_peaks = []

        radCen = int(rads[ringNr])

        # number of connected areas in this ring (less than maxPeaksRing)
        nSegmentsInRing = np.random.randint(0, maxSegmentsInRing)

        # Generate random segments for the ring
        segments = generate_segments(nSegmentsInRing, total_length=nPxEta, min_gap=segDistance,
                                     min_segment_length=minSegWidth, max_segment_length=maxSegWidth)

        segmend_idx = 0
        for segment in segments:
            segmend_idx += 1
            etaCen = int(segment[0] + (segment[1] - segment[0]) / 2)
            etaWidth = segment[1] - segment[0]

            numPeaks = np.random.randint(2, maxNPeaksInSegment)  # Define the num of peaks in the connected area
            # print(f"connect area center position: {etaCen} - connected area width: {etaWidth} - number of peaks: {numPeaks}")

            for peakNr in range(numPeaks):
                # print(f"Total {numPeaks} peaks and Peak index: {peakNr} in segment {segmend_idx}")
                peakCenRad = (radCen + np.random.random(1) * rWidth).item()  # y position of the peak
                peakCenEta = (etaCen + np.random.random(1) * etaWidth).item()  # x position of the peak
                rWidthPeak = 1 + np.random.random(1).item() * (sigmaRMax - 1)
                etaWidthPeak = 2 + np.random.random(1).item() * (sigmaEtaMax - 1)
                x = np.linspace(-int(nSigmas * ceil(rWidthPeak)), int(nSigmas * ceil(rWidthPeak)), endpoint=True,
                                num=(2 * int(nSigmas * ceil(rWidthPeak)) + 1))
                y = np.linspace(-int(nSigmas * ceil(etaWidthPeak)), int(nSigmas * ceil(etaWidthPeak)), endpoint=True,
                                num=(2 * int(nSigmas * ceil(etaWidthPeak)) + 1))
                X, Y = np.meshgrid(x, y)
                Z = gaussian_2d(X, Y, sigma_x=rWidthPeak, sigma_y=etaWidthPeak,
                                intensity=np.random.randint(maxIntensity))

                xStart = int(peakCenRad) - int(nSigmas * ceil(rWidthPeak))
                yStart = int(peakCenEta) - int(nSigmas * ceil(etaWidthPeak))
                if xStart < 0: continue
                if yStart < 0: continue
                if xStart + x.shape[0] > nPxRad: continue
                if yStart + y.shape[0] > nPxEta: continue

                # peak_img = img[xStart:xStart+x.shape[0],yStart:yStart+y.shape[0]]
                # peak_img += np.transpose(Z).astype(np.uint16)

                img[xStart:xStart + x.shape[0], yStart:yStart + y.shape[0]] += np.transpose(Z).astype(np.uint16)
                tmp_mask[xStart:xStart + x.shape[0], yStart:yStart + y.shape[0]] += np.transpose(Z).astype(np.uint16)

                peakPositions.append([peakCenRad, peakCenEta])
                tmp_peaks.append(peakCenEta)
                # print(f"Peak center position: {peakCenEta} - {peakCenRad}")

        #mask = np.log1p(tmp_mask).astype(np.uint16)

        # ring image
        ring_width = 100
        #ring_width = 14
        ring_img = tmp_mask[int(radCen - ring_width / 2): int(radCen + ring_width / 2), :]
        ring_img = np.log1p(ring_img).astype(np.uint16)
        #plt.imshow(ring_img)
        #plt.show()
        continuous_regions, masked_ring_img, instance_idx = find_and_modify_continuous_regions(ring_img, tmp_peaks, instance_idx)
        #abc = find_and_modify_continuous_regions(ring_img)
        # print(continuous_regions) # segments tuple list [(168, 246), (390, 458), (528, 582), (589, 637), (740, 832), (839, 946), (1051, 1091)]

        #plt.imshow(masked_ring_img)
        #plt.show()
        mask[int(radCen - ring_width / 2): int(radCen + ring_width / 2), :] += masked_ring_img
        #plt.imshow(mask)

    max_value = img.max()
    if max_value == 0:
        img_normalized = img.astype(float)
    else:
        img_normalized = img.astype(float) / max_value

    #plt.xlabel('Eta')
    #plt.ylabel('Rad')

    # plot image and mask in the same figure
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(img)
    axs[0].set_title('Image')
    axs[0].set_xlabel('Eta')
    axs[0].set_ylabel('Rad')

    axs[1].imshow(mask)
    axs[1].set_title('Mask')
    axs[1].set_xlabel('Eta')
    axs[1].set_ylabel('Rad')
    
    plt.imsave(f'./tune_overlapping_images/img_{image_idx}.png',img)
    plt.imsave(f'./tune_overlapping_images/mask_{image_idx}.png',mask)
    plt.show()

    #np.save(f'data/npy/hedm_powder/imgs/img_{image_idx}.npy', img)
    #np.save(f'data/npy/hedm_powder/gts/img_{image_idx}.npy', mask)

    #np.save(f'./img_{image_idx}.npy', img)
    #np.save(f'./mask_{image_idx}.npy', mask)

    #plt.savefig(f'./images/sam_size/image_mask_pair_examples/img_mask_{image_idx}.png')


    #plt.imshow(img)
    #plt.savefig(f'img_{image_idx}.png')
    #plt.show()

    #plt.imshow(mask)
    #plt.show()
    #plt.savefig(f'mask_{image_idx}_coor.png')

    #plt.colorbar()

    #plt.imsave(f'img_{image_idx}_coor.png',img)
    #plt.imsave(f'mask_{image_idx}_coor.png',mask)

    #plt.savefig(f'img_{image_idx}_coor.png')
    #plt.savefig(f'mask_{image_idx}_coor.png')

    #plt.imshow(img_normalized)
    #plt.imshow(mask)
    #plt.show()

    #peakPositions = np.array(peakPositions)

    #plt.imsave('output.png',img, cmap='gray', format='png')
    #plt.imshow(np.log(img))
    #plt.imsave('output_mask.png',img)
    # plt.scatter(peakPositions[:,1],peakPositions[:,0])
    #plt.show()

# python main function
def main():
    parser = argparse.ArgumentParser(description='Generate simulation images')
    args = parser.parse_args()

    for i in range(0, 3):
        generate_simulation_image(i)

if __name__ == "__main__":
    main()
