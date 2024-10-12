## Origin https://github.com/MeetKai/functionary/tree/main/functionary/train/packing
## Customize and add FFD packing algorithm

import torch, os
import torch.nn.functional as F
import transformers

patch_visited = {}
booster = os.getenv("ZIN_BOOSTER", "None")
packed_patch_check = os.getenv("PACKED_PATCH_CHECK",0) == "1"
###
def get_max_seqlen_in_batch(attention_mask):
    if packed_patch_check:
        global patch_visited
        booster = os.getenv("ZIN_BOOSTER", "None")
        if booster not in patch_visited:
            patch_visited[booster] = True
            print(f"\033[33m!!! PACKED_PATCH_CHECK=1")
            print(f"!!! booster \033[36m{booster}\033[33m, attention_mask.shape\033[0m", attention_mask.shape)
            # print(f"!!! press enter to continue ... \033[0m", end=""); input()

    max_num = torch.max(attention_mask) # attention_mask: B x N
    counts = [torch.sum(attention_mask == i, axis=-1) \
        for i in range(1, max_num + 1)] # shape: B, count length of data point maksed with i
    result = torch.stack(counts, axis=1)
    result = result.flatten()
    return result[result.nonzero()].squeeze(-1).to(dtype=torch.int32)

def get_unpad_data(attention_mask):
    seqlens_in_batch = get_max_seqlen_in_batch(attention_mask) # attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch

def monkey_patch(caller=None):
    # Monkey-patch flash attention if this transformers already merged: 
    # https://github.com/huggingface/transformers/commit/e314395277d784a34ee99526f48155d4d62cff3d
    # this will work for all models using **flash attention**: Llama, Mistral, Qwen2, Phi3, ...
    if hasattr(transformers, "modeling_flash_attention_utils"):
        already_patched = ( transformers.modeling_flash_attention_utils._get_unpad_data == get_unpad_data )
        if already_patched: print(f"{caller} \033[36mpacked_dataset already patched \033[0m"); return
        transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
    else:  # if this is the old version of transformer
        assert False, "Upgrade transformers để dùng được multi packing dataset"

    print("\033[36mBạn đang sử dụng multi packing, hãy bật biến môi trường PACKED_PATCH_CHECK=1")
    print("để kiểm tra xem việc patching đã thành công chưa.\033[0m")
    print(f"\033[91m!!! PACKED_PATCH_CHECK=1")
    print(f"!!! booster \033[36m{{liger|unsloth|...}}\033[91m, attention_mask.shape ... \033[0m\n")

#################################################################################

from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset

def _ffd_pack_data_points_by_length(offset_lengths, max_length: int) -> List[List[int]]:
    """Pack data points into groups (each group is a new data point), will be used by PackedDataset, 
    to reduce number of data points in training.

    Given lengths of data points, we pack them into groups such that the sum of lengths
    in each group is less than max_length. Each group will be considered as a data point (packed data point)

    This is known as: https://en.wikipedia.org/wiki/Bin_packing_problem
    Args:
        offset_lengths được tách thành offset và lengths
        offset (int): tham số hỗ trợ để xử lý song song nhiều chunks of lengths
        lengths (List[int]): độ dài của các samples
        max_length (int): chính là context length

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...] sau khi đã cộng với offset
    """
    def binary_search(values, low, high, target):
        while low <= high:
            mid = (low + high) // 2
            mid_value = values[mid][1]
            if mid_value == target:
                while mid > 0 and values[mid][1] == target: mid -= 1 # Không bỏ sót
                return mid  # Target found, return the index
            elif mid_value > target: low = mid + 1  # Adjust the search range to the right half
            else: high = mid - 1  # Adjust the search range to the left half
        return 0 if high == 0 else high - 1 # Không bỏ sót

    groups = []
    current_packed_length = 0
    current_group = []

    offset, lengths = offset_lengths
    n = len(lengths)
    selected = [False]*n

    index_length_array = [ (i, lengths[i]) for i in range(n) ]
    index_length_desc = sorted(index_length_array, key=lambda x: -x[1])

    pre_groups = []; maxx = int(os.getenv("MAXX", max_length + 1))#; print(">>> MAXX", maxx)
    for i, l in index_length_desc:
        if l == maxx:
            pre_groups.append([i])
            selected[i] = True

    # https://en.wikipedia.org/wiki/First-fit-decreasing_bin_packing
    # - While there are remaining items:
    #   - Open a new empty bin.
    #   - For each item from largest to smallest:
    #       - If it can fit into the current bin, insert it.
    while True:
        # Dùng binary search để tăng tốc độ tìm kiếm index i thỏa mãn len(item[i]) most likely fitable into current bin
        i = binary_search(index_length_desc, 0, n-1, max_length - current_packed_length)
        while i < n:
            current_index, current_length = index_length_desc[i]
            if selected[current_index] or current_length + current_packed_length > max_length:
                i += 1 # i tìm đc từ binary search rất gần nhưng có thể không thỏa mãn nên tăng dần i lên
            else: # Add to current bin if not selected and fitable into current bin
                current_packed_length += current_length
                current_group.append(current_index)
                selected[current_index] = True
                i = binary_search(index_length_desc, i, n-1, max_length - current_packed_length)

        # Thoát nếu không tạo thêm được group (bin) mới
        if len(current_group) == 0: break

        # Ghi nhận current_group và tạo group mới trống (new empty bin)
        groups.append(current_group)
        if len(pre_groups) > 0:
            current_group = pre_groups.pop()
            current_packed_length = maxx
        else:
            current_group = []
            current_packed_length = 0

    assert len(pre_groups) == 0
    # Đảm bảo không bỏ sót
    assert False not in selected
    # missed_count = 0
    # for i in range(n):
    #     if not selected[i]: groups.append([i]); missed_count += 1
    # if missed_count > 0: print(f">>> offset: {offset}, missed: {missed_count}")
    # Đảm bảo thuật toán chạy đúng
    n = len(lengths)
    total = 0; indexes = set(range(n))
    for group in groups:
        group_length = 0
        for i, x in enumerate(group):
            assert x in indexes
            indexes.remove(x)
            group_length += lengths[x]
            group[i] += offset # điều chỉnh offset để khi nối vào ra kết quả đúng
        assert group_length <= max_length
        total += len(group)

    assert total == n
    assert len(indexes) == 0
    # Rồi mới trả về kết quả
    return groups


from multiprocessing import Pool
from functools import partial
import os, time

def pack_data_points_by_length(
    lengths: List[int], max_length: int, max_size: int = -1
) -> List[List[int]]:
    """given lengths of data points, we merge consecutive data points into a new data point, as long as the concatenated length is less than max_length
    Args:
        lengths (List[int]): List of lengths of data points
        max_length (int): the concatenated length must be less than or equal max_length
        max_size: if != -1; the maximum number of consecutive items being merged; max_size: -1 --> no limit for number of items being merged

    max_size: the maximum number of data points being merged
    For example, lengths=[1, 3, 2, 2, 6, 4, 2, 6, 5]; max_length=10
    if max_size=-1 --> [[0,1,2,3], [4, 5], [6,7], [8]]
    if max_size=3 --> [[0,1,2], [3,4], [5, 6], [7], [8]]

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    result = []
    current_concatenated_length = 0
    current_list = []
    for i in range(len(lengths)):
        cur_length = lengths[i]
        if cur_length + current_concatenated_length <= max_length and (
            max_size == -1 or len(current_list) < max_size
        ):
            current_concatenated_length += cur_length
            current_list.append(i)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(indices) for indices in result]) == len(lengths)
    return result


def ffd_pack_data_points_by_length(lengths: List[int], max_length: int) -> List[List[int]]:
    # Chia để trị và song song hóa
    start_time = time.time()
    print("\n>>> Packing ...")

    # Vì thuật toán ffd tự cài có độ phức tạo ~O(n^2) nên số data point càng to chạy càng lâu
    # Khắc phục: nhận thấy dữ liệu phân bố khá đồng đều theo lengths nên ta chia dữ liệu ra nhiều chunk
    # mỗi chunk có độ dài 40k items để giới hạn thời gian chạy thuật toán ffd, đồng thời dùng nhiều proceeses để cùng xử lý
    # => Kết quả là hầu như chỉ mất dưới 30s để packing hết số lượng raw data 
    chunk_size = 40000
    chunks = [lengths[i:i + chunk_size] for i in range(0, len(lengths), chunk_size)]
    offsets_chunks = [ (idx * chunk_size, chunk) for idx, chunk in enumerate(chunks) ]
    print(">>> Packing chunks:", len(chunks))

    num_proc = os.cpu_count() - 2
    partial_process = partial(_ffd_pack_data_points_by_length, max_length=max_length)

    groups = []
    with Pool(processes=num_proc) as pool:
        for x in pool.imap_unordered(partial_process, offsets_chunks):
            groups += x

    print(f">>> Packing time: {time.time() - start_time} seconds.\n")
    return groups


class PackedDataset(Dataset):
    def __init__(self, dataset: Dataset, tokenizer: Any, pack_length: int, return_tensor=False) -> None:
        super().__init__()
        self.pack_length = pack_length
        self.tokenizer = tokenizer
        self.return_tensor = return_tensor
        self.data_points = dataset
        self.lengths = [ len(x["input_ids"]) for x in dataset ]

        max_len = max(self.lengths)
        assert self.pack_length >= max_len, \
            f"pack_length must be >= max(input lengths), found pack_length={self.pack_length}, max_len={max_len}"

        self.groups = pack_data_points_by_length(self.lengths, self.pack_length)

        # Test lần cuối để đảm bảo thuật toán packing chạy đúng
        total = 0
        for group in self.groups:
            group_length = 0
            for i in group: group_length += self.lengths[i]
            assert group_length <= self.pack_length
            total += len(group)
        assert total == len(self.lengths)


    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_ids = []
        label_ids = []
        attention_mask = []
        position_ids = []

        for index, data_idx in enumerate(self.groups[i]):
            data = self.data_points[data_idx]
            input_ids += data["input_ids"]
            label_ids += [-100] + data["labels"][1:] # ensure the first token won't be included in computing loss

            n = len(data["input_ids"])
            # attention_mask +=         [1]*n # => [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
            attention_mask += [index + 1]*n # => [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 0, 0]
            position_ids += list(range(n))  # => [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 0, 0]

        assert self.tokenizer.padding_side == "right", "Hiện chỉ support padding_side là right"
        pad_leng = self.pack_length - len(input_ids)  # padding to model_max_length
        input_ids = input_ids + [self.tokenizer.pad_token_id]*pad_leng
        label_ids = label_ids + [-100]*pad_leng
        attention_mask = attention_mask + [0]*pad_leng
        position_ids = position_ids + [0]*pad_leng
        assert len(input_ids) == len(label_ids) == len(attention_mask) == self.pack_length

        r = dict(input_ids=input_ids, labels=label_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.return_tensor: r = { k: torch.tensor(v) for k, v in r.items() }
        return r


    def stat(self):
        print(f"\nnumber of original data points: {len(self.data_points)}; \
packed to: {len(self.groups)} data points (x{len(self.data_points)/len(self.groups):0.1f} efficient)")
        original_avg_length = round( sum(self.lengths) / len(self.lengths) )

        total_packed_lengths = sum([ self.lengths[index] for group in self.groups for index in group ])
        avg_packed_length = round( total_packed_lengths / len(self.groups) )

        original_ratio = original_avg_length / self.pack_length
        packed_ratio = avg_packed_length / self.pack_length

        print(f"original avg length: {original_avg_length}/{self.pack_length} ({original_ratio*100:2.1f}%); \
avg packed length: {avg_packed_length}/{self.pack_length} ({packed_ratio*100:2.1f}%)\n")
