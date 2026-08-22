import numpy as np
import os
import glob
import torch
from collections import defaultdict


def read_data(dataset, idx, is_train=True):
    if is_train:
        data_dir = os.path.join('../dataset', dataset, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, 'test/')

    file = data_dir + str(idx) + '.npz'
    with open(file, 'rb') as f:
        data = np.load(f, allow_pickle=True)['data'].tolist()
    return data


# ---------------------------------------------------------------------------
# IoV data: locate once, load once, keep in RAM.
#
# The global test set is a single 2.9 GB file shared by every client. Loading
# it inside read_client_data meant one torch.load per client per round, i.e.
# ~294 GB read from the (network-mounted, read-only) Kaggle input directory
# every round. Caching makes that exactly one load for the whole run.
# ---------------------------------------------------------------------------

_IOV_CACHE = {}
_IOV_BASE_PATH = None

# Tried in order. Set the IOV_DATA_DIR environment variable to override.
_IOV_PATH_CANDIDATES = [
    "/kaggle/input/datasets/tongxuanvu/ids-iov",
    "/kaggle/input/ids-iov",
    "/kaggle/working/ids-iov",
    "D:/FL/data/iov",
    "C:/FederatedLearning/data/iov",
    "C:/FederatedLearning/FL/core/data iov",
]

# Stop caching *train* shards once the cache passes this many bytes, so a
# session cannot be killed by running out of RAM. The test set is always
# cached: it is the one that was being re-read 100 times per round.
_IOV_CACHE_BUDGET = float(os.environ.get("IOV_CACHE_BUDGET_GB", "12")) * (1024 ** 3)


def _tensor_bytes(t):
    return t.numel() * t.element_size()


def _cache_bytes():
    return sum(_tensor_bytes(x) + _tensor_bytes(y) for x, y in _IOV_CACHE.values())


def iov_base_path():
    """Resolve the IoV data root once, and fail loudly if it is missing."""
    global _IOV_BASE_PATH
    if _IOV_BASE_PATH is not None:
        return _IOV_BASE_PATH

    env = os.environ.get("IOV_DATA_DIR")
    for path in ([env] if env else []) + _IOV_PATH_CANDIDATES:
        if path and os.path.isdir(path):
            _IOV_BASE_PATH = path
            print(f"[IoV] data root: {path}", flush=True)
            return path

    # Kaggle mounts datasets under an unpredictable slug; find it by content.
    for root in sorted(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")):
        if os.path.isfile(os.path.join(root, "global_test_data.pt")):
            _IOV_BASE_PATH = root
            print(f"[IoV] data root (auto-detected): {root}", flush=True)
            return root

    raise FileNotFoundError(
        "Khong tim thay thu muc du lieu IoV. Da thu:\n  "
        + "\n  ".join(p for p in ([env] if env else []) + _IOV_PATH_CANDIDATES if p)
        + "\nDat bien moi truong IOV_DATA_DIR tro toi thu muc chua "
          "global_test_data.pt va federated_data/."
    )


def _load_iov(key, pt_path, cacheable=True):
    """Return (x, y) for one IoV file, from cache when possible.

    x is kept in its stored dtype when that dtype is float16 -- the cast to
    float32 happens per batch in TensorBatches, which halves the resident
    size of the 42M-row test set (2.6 GB instead of 5.2 GB).
    """
    hit = _IOV_CACHE.get(key)
    if hit is not None:
        return hit

    try:
        blob = torch.load(pt_path, map_location="cpu")
    except Exception:
        # torch >= 2.6 defaults to weights_only=True; older pickles need False.
        blob = torch.load(pt_path, map_location="cpu", weights_only=False)

    x = blob["x"]
    y = blob["y"].long()
    if x.dtype not in (torch.float16, torch.float32):
        x = x.float()

    if cacheable and _cache_bytes() + _tensor_bytes(x) + _tensor_bytes(y) <= _IOV_CACHE_BUDGET:
        _IOV_CACHE[key] = (x, y)
        print(f"[IoV] cached {key}: {tuple(x.shape)} {x.dtype}, "
              f"cache = {_cache_bytes() / 1024 ** 3:.2f} GB", flush=True)
    elif cacheable:
        print(f"[IoV] cache budget reached ({_IOV_CACHE_BUDGET / 1024 ** 3:.1f} GB); "
              f"{key} will be re-read each time. Raise IOV_CACHE_BUDGET_GB if RAM allows.",
              flush=True)

    return x, y


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    if dataset == "IoV":
        base_path = iov_base_path()
        if is_train:
            x, y = _load_iov(("train", idx), f"{base_path}/federated_data/client_{idx}.pt")
        else:
            # Every client is evaluated on the same global test set, so it is
            # cached under a single key rather than once per client.
            x, y = _load_iov(("test",), f"{base_path}/global_test_data.pt")
        return torch.utils.data.TensorDataset(x, y)

    data = read_data(dataset, idx, is_train)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]
