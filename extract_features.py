import argparse

import numpy as np
import time

import imageio
# import cv2

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from tqdm import tqdm

import glob

import scipy
import scipy.io
import scipy.misc

from lib.model_test import RaPNet
from lib.utils import preprocess_image

from lib.exceptions import EmptyTensorError
from lib.utils import interpolate_dense_features, upscale_positions
from lib.nms import nms_point

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--image_list', type=str, required=True,
    help='path to a list of images to process'
)
parser.add_argument(
    '--file_type', type=str, required=True,
    help='image file suffix'
)
parser.add_argument(
    '--preprocessing', type=str, default='torch',
    help='image preprocessing (caffe or torch)'
)
parser.add_argument(
    '--model_file', type=str, default='models/rapnet.overall.pth',
    help='path to the full model'
)

parser.add_argument(
    '--max_edge', type=int, default=640,
    help='maximum image size at network input'
)
parser.add_argument(
    '--max_sum_edges', type=int, default=1500,
    help='maximum sum of image sizes at network input'
)

parser.add_argument(
    '--output_extension', type=str, default='.rap',
    help='extension for the output'
)
parser.add_argument(
    '--output_type', type=str, default='npz',
    help='output file type (npz or mat)'
)

parser.add_argument(
    '--no-relu', dest='use_relu', action='store_false',
    help='remove ReLU after the dense feature extraction module'
)
parser.set_defaults(use_relu=True)

args = parser.parse_args()

print(args)

# Creating CNN model
model = RaPNet(
    model_file=args.model_file,
    use_relu=True
)
model.to(device)
model.eval()

imgs = sorted(glob.glob('%s/*.%s' % (args.image_list, args.file_type)))
for path in tqdm(imgs, total=len(imgs)):

    image = imageio.imread(path)
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_edge / max(resized_image.shape)
        ).astype('float')
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_sum_edges / sum(resized_image.shape[: 2])
        ).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]


    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )
    with torch.no_grad():
        
        image = torch.tensor(input_image[np.newaxis, :, :, :].astype(np.float32)).to(device)
        b, _, h_init, w_init = image.size()
        device = image.device
        assert(b == 1)

        dense_features = model.dense_feature_extraction(image)

        _, _, h, w = dense_features.size()
        assert h == h_init and w == w_init

        # Recover detections.
        detections = model.detection(dense_features)
        fmap_pos = torch.nonzero(detections[0].cpu()).t()
        del detections

        fmap_keypoints = fmap_pos[1 :, :].float()
        
        try:
            raw_descriptors, _, ids = interpolate_dense_features(
                fmap_keypoints.to(device),
                dense_features[0]
            )
        except EmptyTensorError:
            pass
        fmap_pos = fmap_pos[:, ids]
        fmap_keypoints = fmap_keypoints[:, ids]
        del ids

        # recover keypoints to the same resolution with current input image.
        keypoints = upscale_positions(fmap_keypoints, scaling_steps=0)
        del fmap_keypoints

        # extract attention map
        attention = torch.nn.functional.relu(model.attention(dense_features).squeeze(1))
        attention = attention.cpu()

        descriptors = F.normalize(raw_descriptors, dim=0).cpu()
        del raw_descriptors

        fmap_pos = fmap_pos.cpu()
        keypoints = keypoints.cpu()

        keypoints = torch.cat([
            keypoints,
            torch.ones([1, keypoints.size(1)]) ,
        ], dim=0)

        scores = dense_features[
            0, fmap_pos[0, :], fmap_pos[1, :], fmap_pos[2, :]
        ].cpu() 
        del fmap_pos

        position = keypoints.type(torch.long)
        attentions = attention[0, position[0, :], position[1, :]]
        del position

        keypoints = keypoints.t().numpy()
        scores = scores.numpy()
        descriptors = descriptors.t().numpy()
        attentions = attentions.numpy()

    keypoints, scores, keep_idx = nms_point(keypoints, scores, h_init, w_init, dist_thresh=4)

    descriptors = descriptors[keep_idx]
    attentions = attentions[keep_idx]

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0]]

    """ our(P+R) """

    # partly reweight 
    scores = scores / np.sum(scores)
    thre = np.mean(attentions)
    scores *= np.exp(attentions - thre)

    # select top_k for scores
    location = np.argsort(-scores)[:520]
    keypoints = keypoints[location]
    descriptors = descriptors[location]
    attentions = attentions[location]
    scores = scores[location]

    img_index = path.split('/')[-1]

    # if args.output_type == 'npz':
    #     with open(path + args.output_extension, 'wb') as output_file:
    #         np.savez(
    #             output_file,
    #             keypoints=keypoints,
    #             scores=scores,
    #             descriptors=descriptors,
    #         )
    # elif args.output_type == 'mat':
    #     with open(path + args.output_extension, 'wb') as output_file:
    #         scipy.io.savemat(
    #             output_file,
    #             {
    #                 'keypoints': keypoints,
    #                 'scores': scores,
    #                 'descriptors': descriptors,
    #             }
    #         )
    # else:
    #     raise ValueError('Unknown output type.')
