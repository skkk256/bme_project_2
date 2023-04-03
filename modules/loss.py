import torch.nn as nn


class MyBinaryCrossEntropy(object):
    def __init__(self):
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss(reduction='mean')

    def __call__(self, pred_seg, seg_gt):
        """
        :param pred_seg: [B, C=1, H, W]
        :param seg_gt: [B, C=1, H, W]
        """
        # get the probability by sigmoid. Use BCEloss.
        ######################## WRITE YOUR ANSWER BELOW ########################
        pred_seg_probs = self.sigmoid(pred_seg)
        loss = self.bce(pred_seg_probs, seg_gt)
        #########################################################################
        return loss
