import torch
from torch import Tensor


def top_pool(x:Tensor, dim=1, keep_num:int=None, keep_rate=None, alpha1=1, alpha2=0, exclude_first=True, **kwargs):
    """
    根据输入x 的值和方差来选择 topk 个元素，并返回其 index, index的shape为[B, keep_num, dim]
    选择标准为 alpha1 * mean + alpha2 * std
    args:
        exclude_first: if set to True, will return the index of the first element and the top k-1 elements
    """
    # print('random weight')
    # x = torch.rand(x.shape, device=x.device)
    assert x.ndim == 3, 'input x must have 3 dimensions(B, N, C)'
    assert not (keep_num is not None and keep_rate is not None), 'keep_num and keep_rate can not be assigned on the same time'
    assert not (keep_num is None and keep_rate is None)
    B, N, C = x.shape
    if exclude_first is True:
        x = x[:, 1:, :]
        N -= 1
    if keep_num is None:
        keep_num = max(int(N * keep_rate), 1)
    
    if N == keep_num:
        return None

    mean_weight = x.mean(dim=-1)
    if C == 1:
        std_weight = torch.zeros((B, N)).to(mean_weight.device)
    else:
        std_weight = x.std(dim=-1)
    pool_weight = alpha1 * mean_weight + alpha2 * std_weight
    pool_weight = pool_weight.unsqueeze(-1).expand(B, N, dim)

    if exclude_first is False:
        try:
            _, keep_index = torch.topk(pool_weight, k=keep_num, dim=1, sorted=False)
        except Exception as e:
            print(e)
            print('pool_weight', pool_weight.shape)
            print('k', keep_num)
            exit()
        keep_index, _ = torch.sort(keep_index, dim=1)
    else:
        # pool_weight = pool_weight[:, 1:, ...]
        _, keep_index = torch.topk(pool_weight, k=keep_num, dim=1, sorted=False)
        keep_index, _ = torch.sort(keep_index, dim=1)
        keep_index = torch.cat([torch.zeros([B, 1, dim]).type(torch.int16).to(keep_index.device), keep_index + 1], dim=1)
    return keep_index


if __name__ == '__main__':
    a = torch.tensor([1, 2, 3, 4]).view(1, 4, 1).type(torch.float32)
    print(a.shape)
    print(sparse_pool(a, 2, 1, 0, True))

