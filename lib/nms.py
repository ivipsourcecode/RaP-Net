import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

def nms_point(kpts, scores, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression to selected point. 

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

    Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
        kpts - Nx2 numpy array with keypoint candidate [x_i, y_i].
        scores - Nx1 numpy array with D2 response [score]
        H - Image height.
        W - Image width.
        dist_thresh - Distance to suppress, measured as an infinty norm distance.

    Returns
        nmsed_kpts - Nx2 numpy array with surviving corners.
        nmsed_scores - N length numpy vector with surviving response.
        nmsed_inds - N length numpy vector with surviving corners index.

    """

    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.

    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-scores)
    corners = kpts[inds1]
    values = scores[inds1]
    rcorners = np.floor(corners).astype(int) # Rounded corners.

    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[0] == 0:
        return np.zeros((0, 2)).astype(int), np.zeros(0).astype(int), 0
    if rcorners.shape[0] == 1:
        return kpts, scores, np.arange(0, len(kpts))

    # Initialize the grid.
    for i, rc in enumerate(rcorners):
        grid[rcorners[i, 0], rcorners[i, 1]] = 1
        inds[rcorners[i, 0], rcorners[i, 1]] = i

    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')

    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners):
        # Account for center.
        pt = (rc[0]+pad, rc[1]+pad) #(i, j)

        if grid[pt[0], pt[1]] == 1: # If not yet suppressed.
            grid[pt[0]-pad:pt[0]+pad+1, pt[1]-pad:pt[1]+pad+1] = 0
            grid[pt[0], pt[1]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[inds_keep]
    values = values[inds_keep]

    inds2 = np.argsort(-values)

    nmsed_kpts = out[inds2]
    nmsed_scores = values[inds2]
    nmsed_inds = inds1[inds_keep[inds2]]

    assert len(nmsed_kpts) == len(nmsed_scores) == len(nmsed_inds)

    return nmsed_kpts, nmsed_scores, nmsed_inds
