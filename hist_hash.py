
import torch
import numpy as np
from scipy.sparse import csr_matrix
import cv2
import torch.nn as nn
class HistHash(object):
    def __init__(self, device, pca_num_filter2, hist_blk_size, hist_blk_over):
        self.pca_num_filter2 = pca_num_filter2
        self.hist_blk_size = hist_blk_size
        self.hist_blk_over = hist_blk_over
        self.device = device
    def __call__(self, data):
        map_weights = torch.pow(2.0, torch.arange(
            self.pca_num_filter2-1, -1, -1)).to(self.device)
        stride = round((1-self.hist_blk_over)*self.hist_blk_size)
        unfold = torch.nn.Unfold(kernel_size=self.hist_blk_size, stride=stride)

        im_map = torch.heaviside(data, torch.tensor(0.0))
        #im_map = nn.GELU()(data)
        im_map = im_map.reshape(im_map.shape[0], int(
            im_map.shape[1]/self.pca_num_filter2), self.pca_num_filter2, im_map.shape[2], im_map.shape[3])
        im_map = map_weights[:, None, None] * im_map
        im_map = torch.sum(im_map, dim=2)
        #im_map = self.pool1(im_map)
        patches = unfold(im_map).squeeze(0)
        patches = patches[None, :, :] if patches.ndim==2 else patches
        patches = patches.permute(0, 2, 1)
        patches = patches.reshape([patches.shape[0], self.pca_num_filter2 * patches.shape[1],
                                   self.hist_blk_size*self.hist_blk_size])
        
        codes = torch.tile(torch.arange(0, 2**self.pca_num_filter2, 1), [patches.shape[2],1]).to(self.device)
        features = torch.sum((codes.T[:, None, :] - patches[:, None, :])==0, dim=3)

        features = features.permute(0, 2, 1).reshape(features.shape[0], features.shape[1]*features.shape[2])

        features = csr_matrix(features.cpu())
        # print(type(features[0]))
        # data = np.eye(256)[features[0].tolist()]
        # features[0]=data
        #print(features.shape)

        return features
