import torch


class TensorBatches:
    """Iterate over two in-RAM tensors in slices, without a DataLoader.

    torch.utils.data.DataLoader calls Dataset.__getitem__ once per *sample*
    and then collates. On a tensor with tens of millions of rows that is tens
    of millions of Python calls per pass, which dominates the runtime even
    though the actual arithmetic is trivial. Slicing the tensors directly
    keeps the work in C.

    ``x`` is kept in whatever dtype it was cached in (float16 for the IoV
    set, halving RAM) and cast to ``dtype`` one batch at a time, so the
    consumer still receives float32 exactly as DataLoader used to deliver it.

    Yields the same (x, y) pairs a DataLoader over TensorDataset(x, y) would,
    so metrics computed from it are identical.
    """

    def __init__(self, x, y, batch_size, shuffle=False, drop_last=False,
                 dtype=torch.float32):
        self.x = x
        self.y = y
        self.batch_size = max(1, int(batch_size))
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dtype = dtype

    def __len__(self):
        n = self.x.shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = self.x.shape[0]
        bs = self.batch_size

        if self.shuffle:
            perm = torch.randperm(n)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                if self.drop_last and idx.numel() < bs:
                    break
                yield self.x[idx].to(self.dtype), self.y[idx]
        else:
            for start in range(0, n, bs):
                stop = min(start + bs, n)
                if self.drop_last and stop - start < bs:
                    break
                yield self.x[start:stop].to(self.dtype), self.y[start:stop]
