from re import T
import torch
from flow_fusion import netfusion

def test_flow_fusion():
    model = netfusion()

    im1 = torch.rand(1, 3, 224, 224)
    im2 = torch.rand(1, 3, 224, 224)
    flow1 = torch.rand(1, 2, 224, 224)
    flow2 = torch.rand(1, 2, 224, 224)
    flow3 = torch.rand(1, 2, 224, 224)

    adapted_flow = model.fuse_flows(im1, im2, flow1, flow2, flow3)
