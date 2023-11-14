import os
import cv2
import pyvips
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from .utils import get_concentrations, percentile, TissueMaskException
from .stain_extractor_cpu import StainExtractorCPU
from .stain_extractor_gpu import StainExtractorGPU
from tqdm import tqdm
from einops import rearrange
from utils.utils import standardize


class TorchVahadaneNormalizer():
    """
    Source code adapted from:
    https://github.com/Peter554/StainTools/blob/master/staintools/stain_normalizer.py
    Uses StainTools as dependency could be changed by integrating VahadaneStainExtractor directly.
    if direct usage on gpu, idea is possibility to use nvidia based loadings with gpu decompression of wsi
    """

    def __init__(self, device='cuda', staintools_estimate=True, verbose=True):
        super().__init__()
        self.stain_matrix_target = None
        self.maxC_target = None
        self.method = 'ista'
        self.device = device  # torch.device(device)
        self.staintools_estimate = staintools_estimate
        self.verbose = verbose
        if self.staintools_estimate:
            self.stain_extractor = StainExtractorCPU()
        else:
            self.stain_extractor = StainExtractorGPU()

    def get_coefs(self, I):

        if self.staintools_estimate:
            stain_matrix = self.stain_extractor.get_stain_matrix(I).astype(np.float32)
            stain_matrix = torch.from_numpy(stain_matrix).to(self.device)
            I = torch.from_numpy(I).to(self.device)
        else:
            if not (type(I) == torch.Tensor):
                I = torch.from_numpy(I).to(self.device)
            stain_matrix = self.stain_extractor.get_stain_matrix(I, device=self.device)

        concentrations = get_concentrations(I, stain_matrix, method=self.method)
        maxC = percentile(concentrations.T, 99, dim=0)
        return stain_matrix.cpu().numpy(), maxC.cpu().numpy()  # type: ignore
    

    def fit(self, dataloader: DataLoader, p, div: int = 8):
        stain_mats = []
        max_cs = []
        for n, (data, coor, size) in tqdm(enumerate(dataloader)):
            if n > (len(dataloader) // div):
                break
            data = rearrange(data, 'n h w d -> (n h) w d')
            coor = coor.numpy()
            size = size.numpy()
            data = data.numpy()
            try:
                data, _ = standardize(data, p=p)
                stain_mat, max_c = self.get_coefs(data)
                stain_mats.append(stain_mat)
                max_cs.append(max_c)
            except TissueMaskException:
                pass
        self.stain_matrix_target = torch.from_numpy(np.median(np.array(stain_mats), axis=0)).to(self.device)
        self.maxC_target = torch.from_numpy(np.median(np.array(max_cs), axis=0)).to(self.device)
        # verbose
        if self.verbose:
            print(self.stain_matrix_target.cpu(), 'set as target stain matrix')
            print('max concentration of target:  ', self.maxC_target.cpu())


    def transform(self, I, stain_mat, max_c):
        # verbose
        if self.verbose:
            print(self.stain_matrix_target, '(target stain matrix)')
            print(stain_mat, 'set as transform matrix')
            print('maxC_target: ', self.maxC_target.cpu())  # type: ignore
            print('maxC:        ', max_c.cpu())
            print('align_ratio: ', (self.maxC_target / max_c).cpu())
            self.verbose = False
        
        if not (type(I) == torch.Tensor):
            I = torch.from_numpy(I).to(self.device)

        concentrations = get_concentrations(I, stain_mat, method=self.method)
        concentrations *= (self.maxC_target / max_c)[:, None]

        out = 255 * torch.exp(-1 * torch.matmul(concentrations.T, self.stain_matrix_target))  # type: ignore
        return out.reshape(I.shape).type(torch.uint8)
