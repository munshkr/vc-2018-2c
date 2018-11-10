#!/usr/bin/env python3
import numpy as np
from imageio import imread, imwrite
from scipy.io import loadmat, savemat

def hist_thresh(infile, outfile, orientation, low, high):
    Im = np.loadtxt(infile)
    Io = np.loadtxt(orientation)

    rows, cols = np.nonzero(Im > high)
    S = list(zip(rows, cols))

    out = np.zeros(Im.shape, dtype=np.uint8)

    while S:
        r, c = S.pop()
        if out[r, c] > 0:
            continue

        # Mark as border
        out[r, c] = 255

        # Convert angle in radians to degrees
        angle = Io[r, c] * 180 / np.pi;

        # Grow region if magnitude exceeds low threshold
        if angle < 45 and angle > -45:
            if Im[r-1, c] > low:
                S.append((r-1, c))
            if Im[r+1, c] > low:
                S.append((r+1, c))
        elif angle > 45 and angle < 135:
            if Im[r-1, c-1] > low:
                S.append((r-1, c-1))
            if Im[r+1, c+1] > low:
                S.append((r+1, c+1))
        elif angle > 135 and angle < -135:
            if Im[r, c-1] > low:
                S.append((r, c-1))
            if Im[r, c+1] > low:
                S.append((r, c+1))
        elif angle > -135 and angle > -45:
            if Im[r-1, c+1] > low:
                S.append((r-1, c+1))
            if Im[r+1, c-1] > low:
                S.append((r+1, c-1))

    imwrite(outfile, out)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
            description='Apply hysteresis thresholding on image',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input',
            help='path to input image file')
    parser.add_argument('output',
            help='path to output image file')

    parser.add_argument('orientation',
            help='path to image of gradient orientation')
    parser.add_argument('low', type=int, help='low threshold')
    parser.add_argument('high', type=int, help='high threshold')

    args = parser.parse_args()

    hist_thresh(args.input, args.output, args.orientation, args.low, args.high)
