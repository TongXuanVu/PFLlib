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
_IOV_TASK = None            # None/0 = gop moi task; 1..5 = chi mot task
_IOV_TEST_VIEW = None       # tap test da loc theo cac lop cua task dang chay
_IOV_TRAIN_LABELS = set()   # cac nhan thuc su xuat hien trong du lieu train
_IOV_EVAL_CAP = 0           # tran so dong test moi lop; 0 = dung toan bo
_IOV_EVAL_W = None          # trong so hoan nguyen cho tung lop sau khi lay mau
_IOV_FED_DIR = "federated_data"   # full | _10shot | _fewshot (1%)
_IOV_EVAL_CUM = False       # danh gia luy tien tren MOI lop da hoc tu task 1
_IOV_TASK_SIZES = None      # so lop moi task, doc tu task_mapping.json

# Tried in order. Set the IOV_DATA_DIR environment variable to override.
_IOV_PATH_CANDIDATES = [
    "/kaggle/input/datasets/tongxuanvu/100clientiov",
    "/kaggle/input/datasets/tongxuanvu/ids-iov",
    "/kaggle/input/ids-iov",
    "D:/FL/data/iov",
    "C:/FederatedLearning/data/iov",
]

_IOV_CACHE_BUDGET = float(os.environ.get("IOV_CACHE_BUDGET_GB", "12")) * (1024 ** 3)


def set_iov_fed_dir(name):
    """Thu muc con chua shard train: federated_data (full),
    federated_data_10shot, federated_data_fewshot (1%)."""
    global _IOV_FED_DIR, _IOV_CACHE
    if name and name != _IOV_FED_DIR:
        _IOV_FED_DIR = name
        _IOV_CACHE = {k: v for k, v in _IOV_CACHE.items() if k == ("test",)}
        print(f"[IoV] thu muc du lieu train: {name}", flush=True)


def _task_sizes():
    """So lop moi task, uu tien doc task_mapping.json cua chinh dataset."""
    global _IOV_TASK_SIZES
    if _IOV_TASK_SIZES is None:
        _IOV_TASK_SIZES = [3, 3, 3, 2, 2]
        f = os.path.join(iov_base_path(), "task_mapping.json")
        if os.path.exists(f):
            try:
                import json
                with open(f, encoding='utf-8') as fh:
                    m = json.load(fh)
                if isinstance(m, list) and all(isinstance(t, list) for t in m):
                    _IOV_TASK_SIZES = [len(t) for t in m]
            except Exception as e:
                print(f"[IoV] khong doc duoc task_mapping.json ({e}), dung [3,3,3,2,2]",
                      flush=True)
        print(f"[IoV] so lop moi task: {_IOV_TASK_SIZES}", flush=True)
    return _IOV_TASK_SIZES


def set_iov_eval_cumulative(flag):
    """True = danh gia tren cac lop cua task 1..tid (quy uoc class-incremental,
    de lo ra viec quen lop cu). False = chi cac lop cua task dang chay."""
    global _IOV_EVAL_CUM, _IOV_TEST_VIEW, _IOV_EVAL_W
    _IOV_EVAL_CUM = bool(flag)
    _IOV_TEST_VIEW = None
    _IOV_EVAL_W = None


def set_iov_eval_cap(cap):
    """Tran so dong test giu lai cho MOI lop. 0 = khong lay mau."""
    global _IOV_EVAL_CAP, _IOV_TEST_VIEW, _IOV_EVAL_W
    _IOV_EVAL_CAP = max(0, int(cap or 0))
    _IOV_TEST_VIEW = None
    _IOV_EVAL_W = None


def iov_eval_weights():
    """Trong so mot dong test da lay mau dai dien cho bao nhieu dong that.

    Lay mau THEO NHAN, nen moi hang cua confusion matrix duoc nhan lai dung mot
    he so -> uoc luong khong chech cua confusion matrix day du. Lop it hon tran
    duoc giu nguyen (he so 1), chi lop da so bi cat.
    """
    return _IOV_EVAL_W


def set_iov_task(task_id):
    """Chon task cho bo IoV class-incremental. 0/None = gop toan bo task."""
    global _IOV_TASK, _IOV_TEST_VIEW, _IOV_TRAIN_LABELS, _IOV_EVAL_W
    _IOV_TASK = int(task_id) if task_id else None
    _IOV_TEST_VIEW = None
    _IOV_EVAL_W = None
    _IOV_TRAIN_LABELS = set()
    if _IOV_TASK:
        print(f"[IoV] chi dung task {_IOV_TASK}", flush=True)


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

    for root in sorted(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")
                       + glob.glob("/kaggle/input/*/*/*")):
        if os.path.isfile(os.path.join(root, "global_test_data.pt")):
            _IOV_BASE_PATH = root
            print(f"[IoV] data root (auto-detected): {root}", flush=True)
            return root

    raise FileNotFoundError(
        "Khong tim thay thu muc du lieu IoV. Dat bien moi truong IOV_DATA_DIR "
        "tro toi thu muc chua global_test_data.pt va federated_data/."
    )


def _client_files(idx):
    """Cac file train cua client idx, theo layout thuc te cua dataset.

    Ho tro ca hai kieu dat ten:
      federated_data/client_<idx>.pt              (khong chia task)
      federated_data/client_<idx>_task_<t>.pt     (class-incremental)
    """
    fed = os.path.join(iov_base_path(), _IOV_FED_DIR)
    flat = os.path.join(fed, f"client_{idx}.pt")
    if _IOV_TASK:
        p = os.path.join(fed, f"client_{idx}_task_{_IOV_TASK}.pt")
        return [p] if os.path.exists(p) else []
    if os.path.exists(flat):
        return [flat]
    return sorted(glob.glob(os.path.join(fed, f"client_{idx}_task_*.pt")))


def iov_clients_available(max_clients=1000):
    """So client co du lieu voi task dang chon -- de bao loi cho ro."""
    n = 0
    for i in range(max_clients):
        if _client_files(i):
            n += 1
        elif n:
            break
    return n


def _read_pt(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _norm(x, y):
    y = y.long()
    if x.dtype not in (torch.float16, torch.float32):
        x = x.float()
    return x, y


def _load_iov_train(idx):
    key = ("train", _IOV_TASK, idx)
    hit = _IOV_CACHE.get(key)
    if hit is not None:
        return hit

    files = _client_files(idx)
    if not files:
        avail = iov_clients_available()
        raise FileNotFoundError(
            f"Client {idx} khong co du lieu train"
            + (f" o task {_IOV_TASK}" if _IOV_TASK else "")
            + f". Chi co {avail} client (id 0..{avail-1}) co du lieu -> dat -nc {avail}."
        )

    xs, ys = [], []
    for f in files:
        blob = _read_pt(f)
        x, y = _norm(blob["x"], blob["y"])
        xs.append(x); ys.append(y)
    x = xs[0] if len(xs) == 1 else torch.cat(xs)
    y = ys[0] if len(ys) == 1 else torch.cat(ys)

    _IOV_TRAIN_LABELS.update(y.unique().tolist())

    if _cache_bytes() + _tensor_bytes(x) + _tensor_bytes(y) <= _IOV_CACHE_BUDGET:
        _IOV_CACHE[key] = (x, y)
    return x, y


def _load_iov_test_full():
    hit = _IOV_CACHE.get(("test",))
    if hit is not None:
        return hit
    blob = _read_pt(os.path.join(iov_base_path(), "global_test_data.pt"))
    x, y = _norm(blob["x"], blob["y"])
    _IOV_CACHE[("test",)] = (x, y)
    print(f"[IoV] cached test: {tuple(x.shape)} {x.dtype}, "
          f"cache = {_cache_bytes() / 1024 ** 3:.2f} GB", flush=True)
    return x, y


def iov_test_view():
    """Tap test dung de danh gia.

    Khi chay mot task cu the, mo hinh chi duoc hoc cac lop cua task do, nen
    danh gia tren ca 13 lop se do mot thu mo hinh chua bao gio duoc day. O day
    tap test duoc loc ve dung cac nhan CO MAT trong du lieu train da nap --
    suy tu chinh du lieu, khong can file mapping. Goi lan dau sau khi toan bo
    client da duoc tao, luc do _IOV_TRAIN_LABELS moi day du.
    """
    global _IOV_TEST_VIEW, _IOV_EVAL_W
    if _IOV_TEST_VIEW is not None:
        return _IOV_TEST_VIEW
    x, y = _load_iov_test_full()
    if _IOV_TASK:
        if _IOV_EVAL_CUM:
            # Luy tien: moi lop cua task 1..tid. Nhan chay lien tuc tu 0 theo
            # dung thu tu task, da doi chieu voi phan bo lop cua tap test.
            cum = sum(_task_sizes()[:_IOV_TASK])
            keep = list(range(cum))
            what = f"luy tien task 1-{_IOV_TASK}"
        elif _IOV_TRAIN_LABELS:
            keep = sorted(_IOV_TRAIN_LABELS)
            what = f"rieng task {_IOV_TASK}"
        else:
            keep = None
        if keep:
            mask = torch.isin(y, torch.tensor(keep, dtype=y.dtype))
            x, y = x[mask], y[mask]
            print(f"[IoV] danh gia {what}: {len(keep)} lop {keep}"
                  f" -> {x.shape[0]:,} dong test (tu {mask.numel():,})", flush=True)

    if _IOV_EVAL_CAP:
        # Lay mau theo tung nhan, hat giong co dinh -> moi round dung DUNG mot
        # tap con, nen duong cong qua cac round van so sanh duoc voi nhau.
        g = torch.Generator().manual_seed(12345)
        n_cls = int(y.max().item()) + 1
        w = torch.ones(n_cls, dtype=torch.float64)
        idx_keep = []
        for c in range(n_cls):
            idx_c = (y == c).nonzero(as_tuple=True)[0]
            n = idx_c.numel()
            if n == 0:
                continue
            if n > _IOV_EVAL_CAP:
                pick = torch.randperm(n, generator=g)[:_IOV_EVAL_CAP]
                idx_c = idx_c[pick]
                w[c] = n / float(_IOV_EVAL_CAP)
            idx_keep.append(idx_c)
        sel = torch.cat(idx_keep).sort().values
        before = y.shape[0]
        x, y = x[sel], y[sel]
        _IOV_EVAL_W = w
        capped = [(c, float(w[c])) for c in range(n_cls) if w[c] > 1]
        print(f"[IoV] lay mau tap test: {before:,} -> {y.shape[0]:,} dong "
              f"(tran {_IOV_EVAL_CAP:,}/lop). He so hoan nguyen: "
              f"{ {c: round(f,1) for c, f in capped} }", flush=True)

    _IOV_TEST_VIEW = (x, y)
    return _IOV_TEST_VIEW


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    if dataset == "IoV":
        if is_train:
            x, y = _load_iov_train(idx)
        else:
            x, y = _load_iov_test_full()
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
