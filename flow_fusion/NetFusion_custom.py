#! /usr/bin/env python

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from pkg_resources import resource_filename

import torch
import torch.nn as nn
from .NetFusion import NetFusion


CHECKPOINT_PATH = resource_filename('flow_fusion', 'checkpoints/fusion_net.pth.tar')

__all__ = [
 'netfusion'
]


def channelnorm(input: torch.Tensor) -> torch.Tensor:
    return torch.norm(input, dim=1, p=2, keepdim=True)


class NetFusion_custom(nn.Module):
    def __init__(self, div_flow = 20.0, batchNorm=False):
        super(NetFusion_custom,self).__init__()

        self.batchNorm = batchNorm
        self.div_flow = div_flow

        self.fusion = NetFusion(batchNorm=self.batchNorm, inPlanes=9)

    def warp_back(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()

        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        vgrid[:,0,:,:] = 2.0 * vgrid[:,0,:,:] / max(W - 1, 1) - 1.0
        vgrid[:,1,:,:] = 2.0 * vgrid[:,1,:,:] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0,2,3,1)
        output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        mask = torch.ones(x.size())
        if x.is_cuda:
            mask = mask.cuda()
        mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

        mask[mask<0.999] = 0
        mask[mask>0] = 1

        return output*mask

    def forward(self, im1, im2, cur_flow, prev_flow, prev_flow_back):
        b, _, h, w = im1.shape
        prev_flow = self.warp_back(prev_flow, prev_flow_back) #warp 0->1 to frame 1

        im2_warp_backward_cur_flow = self.warp_back(im2, cur_flow)  # im2 1->2 im1
        im2_warp_backward_prev_flow = self.warp_back(im2, prev_flow) # im2 wapred 1->2 im1

        mask_warp_cur_flow = torch.ones(b, 1, h, w).float()
        if im1.is_cuda:
            mask_warp_cur_flow = mask_warp_cur_flow.cuda()
        mask_warp_prev_flow = torch.ones(b, 1, h, w).float()
        if im1.is_cuda:
            mask_warp_prev_flow = mask_warp_prev_flow.cuda()

        mask_warp_cur_flow = self.warp_back(mask_warp_cur_flow, cur_flow)
        mask_warp_prev_flow = self.warp_back(mask_warp_prev_flow, prev_flow)

        cur_flow = cur_flow / self.div_flow
        prev_flow = prev_flow / self.div_flow
        
        # norm_cur_flow = channelnorm(cur_flow)
        # norm_prev_flow = channelnorm(prev_flow)

        diff_im1_cur_flow = channelnorm(im1-im2_warp_backward_cur_flow)
        diff_im1_prev_flow = channelnorm(im1-im2_warp_backward_prev_flow)

        diff_im1_cur_flow_comp = 0.5 - diff_im1_cur_flow
        diff_im1_cur_flow_comp[mask_warp_cur_flow>0] = 0
        diff_im1_prev_flow_comp = 0.5 - diff_im1_prev_flow
        diff_im1_prev_flow_comp[mask_warp_prev_flow>0] = 0

        diff_im1_cur_flow = diff_im1_cur_flow + diff_im1_cur_flow_comp
        diff_im1_prev_flow = diff_im1_prev_flow + diff_im1_prev_flow_comp

        concat_feat = torch.cat((im1, cur_flow, prev_flow, diff_im1_cur_flow, diff_im1_prev_flow), dim=1)

        flow_new = self.fusion(concat_feat)

        return flow_new


    def fuse_flows(self, image1, image2, curr_flow, prev_flow12, prev_flow21):
        flow = self(image1, image2, curr_flow, prev_flow12, prev_flow21)
        flow = flow[0] * self.div_flow
        return flow # smoothed curr_flow with regard to previous flows


def netfusion(pretrained: bool = True, div_flow: float = 20.0, batchNorm: bool = False):
    model = NetFusion_custom(div_flow=div_flow, batchNorm=batchNorm)
    if pretrained:
        data = torch.load(CHECKPOINT_PATH)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)

    return model.eval()
