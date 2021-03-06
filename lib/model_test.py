import torch
import torch.nn as nn
import torch.nn.functional as F
# from lib.model import RegionWeight


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_relu=True):
        super(DenseFeatureExtractionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, padding=1, stride=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, padding=1, stride=1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, padding=1, stride=1),
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
        )
        self.num_channels = 512

        self.use_relu = use_relu

    def forward(self, batch):
        output = self.model(batch)
        if self.use_relu:
            output = F.relu(output)
        return output
    
    
class RegionWeight(torch.nn.Module):
    def __init__(self, in_channel):
        super(RegionWeight, self).__init__()
        # branch 0
        self.branch_0 = torch.nn.Conv2d(in_channel, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.norm_0 = torch.nn.BatchNorm2d(32, affine=True)
        # branch 1
        self.branch_1 = torch.nn.Conv2d(in_channel, 32, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.norm_1 = torch.nn.BatchNorm2d(32, affine=True)
        # branch 2
        self.branch_2 = torch.nn.Conv2d(in_channel, 32, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.norm_2 = torch.nn.BatchNorm2d(32, affine=True)
        # activation
        self.relu = torch.nn.ReLU(inplace=True)
        self.softplus = torch.nn.Softplus()
        # score
        self.attention_score = torch.nn.Conv2d(96, 1, kernel_size=(1, 1), stride=1, padding=0)
    
    def forward(self, x):
        # down sampling
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        branch_0 = self.relu(self.norm_0(self.branch_0(x)))
        branch_1 = self.relu(self.norm_1(self.branch_1(x)))
        branch_2 = self.relu(self.norm_2(self.branch_2(x)))

        fusion = torch.cat([branch_0, branch_1, branch_2], 1)

        # score = torch.nn.functional.relu(self.softplus(self.attention_score(fusion)))
        score = self.softplus(self.attention_score(fusion))

        # recover resolution
        score = torch.nn.functional.upsample_nearest(score, scale_factor=2)
        
        return score


class RaPNet(nn.Module):
    def __init__(self, model_file=None, use_relu=True):
        super(RaPNet, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(use_relu=use_relu)

        self.attention = RegionWeight(in_channel=self.dense_feature_extraction.num_channels)

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()

        if model_file is not None:
            checkpoint = torch.load(model_file)['model']
            basenet_weights = self.state_dict()
            load_weights = {k:v for k,v in checkpoint.items() if k in basenet_weights}
            print(load_weights.keys())
            basenet_weights.update(load_weights)
            self.load_state_dict(basenet_weights) 
            del checkpoint
        else:
            raise IOError('CANNOT find %s pre-trained weight!' % model_file)

    def forward(self, batch):
        _, _, h, w = batch.size()
        
        dense_features = self.dense_feature_extraction(batch)

        attention = self.attention(dense_features).squeeze(1)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'attentions': attention,
            'detections': detections,
            'displacements': displacements
        }


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 27, stride=1, padding=13)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)
