import numpy as np
import skimage.morphology
import skimage.measure
import math

def get_terminations_bifurcations(skel, mask):
    """
    Detect termination and bifurcation points in skeletonized fingerprint.
    """
    skel = skel == 255
    (rows, cols) = skel.shape
    minutiae_term = np.zeros(skel.shape)
    minutiae_bif = np.zeros(skel.shape)
    
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if skel[i][j]:
                block = skel[i-1:i+2, j-1:j+2]
                block_val = np.sum(block)
                if block_val == 2:  # Termination (1 pixel with 1 neighbor)
                    minutiae_term[i, j] = 1
                elif block_val == 4:  # Bifurcation (1 pixel with 3 neighbors)
                    minutiae_bif[i, j] = 1
    
    mask = skimage.morphology.convex_hull_image(mask > 0)
    mask = skimage.morphology.erosion(mask, np.ones((5, 5)))
    minutiae_term = np.uint8(mask) * minutiae_term
    minutiae_bif = np.uint8(mask) * minutiae_bif
    
    return minutiae_term, minutiae_bif

class MinutiaeFeature:
    def __init__(self, locX, locY, orientation, type):
        self.locX = locX
        self.locY = locY
        self.orientation = orientation
        self.type = type

def compute_angle(block, minutiae_type):
    """
    Compute orientation angle for minutiae points.
    """
    angle = 0
    (blk_rows, blk_cols) = block.shape
    center_x, center_y = (blk_rows-1)/2, (blk_cols-1)/2
    
    if minutiae_type.lower() == 'termination':
        sum_val = 0
        for i in range(blk_rows):
            for j in range(blk_cols):
                if (i == 0 or i == blk_rows-1 or j == 0 or j == blk_cols-1) and block[i][j] != 0:
                    angle = -math.degrees(math.atan2(i-center_y, j-center_x))
                    sum_val += 1
                    if sum_val > 1:
                        angle = float('nan')
        return angle
    elif minutiae_type.lower() == 'bifurcation':
        angle = []
        sum_val = 0
        for i in range(blk_rows):
            for j in range(blk_cols):
                if (i == 0 or i == blk_rows-1 or j == 0 or j == blk_cols-1) and block[i][j] != 0:
                    angle.append(-math.degrees(math.atan2(i-center_y, j-center_x)))
                    sum_val += 1
        if sum_val != 3:
            angle = float('nan')
        return angle

def extract_minutiae_features(skel, minutiae_term, minutiae_bif):
    """
    Extract minutiae features (location, orientation, type).
    """
    features_term = []
    minutiae_term = skimage.measure.label(minutiae_term, connectivity=2)
    rp = skimage.measure.regionprops(minutiae_term)
    window_size = 2
    
    for i in rp:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row-window_size:row+window_size+1, col-window_size:col+window_size+1]
        angle = compute_angle(block, 'Termination')
        features_term.append(MinutiaeFeature(row, col, angle, 'Termination'))
    
    features_bif = []
    minutiae_bif = skimage.measure.label(minutiae_bif, connectivity=2)
    rp = skimage.measure.regionprops(minutiae_bif)
    window_size = 1
    
    for i in rp:
        (row, col) = np.int16(np.round(i['Centroid']))
        block = skel[row-window_size:row+window_size+1, col-window_size:col+window_size+1]
        angle = compute_angle(block, 'Bifurcation')
        features_bif.append(MinutiaeFeature(row, col, angle, 'Bifurcation'))
    
    return features_term, features_bif