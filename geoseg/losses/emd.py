from typing import List
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def tensor2pointcloud(im: Tensor, norm=True):
    if im.ndim == 4:
        im = im.argmax(dim=1, keepdim=True)
    if im.ndim == 3:
        im = im.unsqueeze(1)

    assert im.ndim == 4, NotImplementedError("ndim of the input must be 4")

    im = F.interpolate(
        im.to(dtype=torch.float32), scale_factor=0.125, mode="nearest"
    ).squeeze(
        1
    )  # avoid OOM ERROR

    # TODO multi-class, multi-channel

    signature = []
    num_points_list = []  # 新增，用于记录每个样本每个类别的点数
    for i in range(im.shape[0]):
        coords = torch.nonzero(im[i] == 1).to(dtype=torch.float32)
        num_points = coords.shape[0]  # 获取当前样本对应的点数
        if norm:
            coords -= coords.mean(dim=0, keepdim=True)
        signature.append(coords)
        num_points_list.append(num_points)

    return signature, num_points_list


def group_samples_by_num_points(num_points_list, intervals: List[tuple]):
    groups = {interval: [] for interval in intervals}
    for idx, num_points_per_sample in enumerate(num_points_list):
        for i, interval in enumerate(intervals):
            if all([interval[0] <= num_points_per_sample <= interval[1]]):
                groups[interval].append(idx)
    return groups


def pad_group(group_sig):
    max_num_points = max([s.shape[0] for s in group_sig])
    padded_sig = []
    mask = []
    for sample in group_sig:
        num_points = sample.shape[0]
        pad_size = max_num_points - num_points
        padded_sample = torch.cat(
            [
                sample,
                torch.zeros(
                    (pad_size, sample.shape[1]),
                    dtype=torch.float32,
                    device=sample.device,
                ),
            ],
            dim=0,
        )
        padded_sig.append(padded_sample)
        sample_mask = torch.cat(
            [
                torch.ones((num_points,), dtype=torch.bool),
                torch.zeros((pad_size,), dtype=torch.bool),
            ],
            dim=0,
        ).to(sample.device)
        mask.append(sample_mask)
    return torch.stack(padded_sig), mask


def unpad_loss(group_loss, logit_mask, target_mask):
    valid_loss = []
    for sample_loss, sample_logit_mask, sample_target_mask in zip(
        group_loss, logit_mask, target_mask
    ):
        valid_sample_loss = sample_loss[sample_logit_mask][:, sample_target_mask]
        valid_loss.append(torch.sum(valid_sample_loss))
    return valid_loss


def diff_num_emdloss(
    logits: Tensor,
    targets: Tensor,
    eps=0.01,
    max_iter=100,
    reduction="none",
):
    batchsize = logits.shape[0]
    logit_sig, logit_num_points = tensor2pointcloud(logits, norm=True)
    target_sig, target_num_points = tensor2pointcloud(targets, norm=True)

    # 设定点数区间
    # intervals = [
    #     (1, 64 * 64),
    #     (64 * 64 + 1, 128 * 128),
    #     (128 * 128 + 1, 256 * 256),
    #     (256 * 256 + 1, 512 * 512),
    # ]
    intervals = [
        (1, 32 * 32),
        (32 * 32 + 1, 64 * 64),
        (64 * 64 + 1, 128 * 128),
    ]
    sample_groups = group_samples_by_num_points(target_num_points, intervals)

    all_losses = []
    for interval, group_indices in sample_groups.items():
        if not group_indices:  # 如果该组没有样本，跳过
            continue
        group_logit_sig = [logit_sig[i] for i in group_indices]
        group_target_sig = [target_sig[i] for i in group_indices]

        # 进行填充等预处理，使组内可以批量计算，以下是简单示意，需根据实际完善
        padded_logit_sig, logit_mask = pad_group(group_logit_sig)
        padded_target_sig, target_mask = pad_group(group_target_sig)

        EMD = EMDLoss(eps, max_iter, reduction="none")
        group_loss, _, _ = EMD(padded_logit_sig, padded_target_sig)

        # 根据填充情况还原损失计算，去除填充部分的影响（示例简化，实际可能更复杂）
        group_loss = unpad_loss(group_loss, logit_mask, target_mask)
        all_losses.extend(group_loss)

    if reduction == "mean":
        return sum(all_losses) / len(all_losses)
    elif reduction == "sum":
        return sum(all_losses)

    return all_losses


class EMDLoss(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=0.01, max_iter=100, reduction="none"):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor):
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y).to(x.device)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = (
            torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / x_points)
            .squeeze()
            .to(x.device)
        )
        nu = (
            torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False)
            .fill_(1.0 / y_points)
            .squeeze()
            .to(x.device)
        )
        print(f"mu tensor dimensions: {mu.dim()}")
        print(f"nu tensor dimensions: {nu.dim()}")

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (
                self.eps
                * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1))
                + u
            )
            v = (
                self.eps
                * (
                    torch.log(nu + 1e-8)
                    - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)
                )
                + v
            )
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        # cost = torch.sum(pi * C, dim=(-2, -1))
        cost = pi * C
        print(f"cost shape: {cost.shape}")

        if self.reduction == "mean":
            cost = torch.sum(cost, dim=(-2, -1))
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
