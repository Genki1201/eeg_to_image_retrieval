import torch
import torch.nn as nn

# コサイン類似度損失
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, input, target):
        # 1 - コサイン類似度を計算
        cosine_sim = self.cosine_similarity(input, target)
        loss = 1 - cosine_sim.mean()  # 平均を取る
        return loss
