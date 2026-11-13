"""Count-sketch of parameter gradients (pilot R012b, R014, B0b).

S(x)[b] = sum_{j: h(j)=b} s(j) x_j over all parameters, D = 2^18 buckets. Hashes are per tensor,
seeded by crc32(name) and the sketch seed, on the element index j *within the original tensor*
(Dietzfelbinger multiply-add-shift on 64-bit words, strongly universal for j < 2^32):
  h(j) = ((a*j + b) mod 2^64) >> (64 - log2 D),   s(j) = +1 / -1 from the top bit of (c*j + d) mod 2^64.
So signs are pairwise independent and E<S(x), S(y)> = <x, y>. Because j is local to the named
tensor, a full replica and an FSDP shard (a slice of the flat parameter) give identical sketches.
torch int64 multiply/add wrap modulo 2^64 (two's complement); scripts/plan/r014_sketch_test.py
checks the hash against exact Python integers.
"""

import zlib

import numpy as np
import torch

D_DEFAULT = 2 ** 18
CHUNK = 1 << 24
_MASK64 = (1 << 64) - 1


def _to_i64(u):
    """Unsigned 64-bit integer -> the int64 with the same bits."""
    return u - (1 << 64) if u >= (1 << 63) else u


def tensor_coeffs(name, seed=0):
    """(a, b, c, d) as unsigned 64-bit Python ints; a and c odd."""
    rng = np.random.RandomState((zlib.crc32(name.encode('utf-8')) ^ (seed * 0x9E3779B1)) & 0xFFFFFFFF)
    words = [int(x) for x in rng.randint(0, 1 << 32, size=8, dtype=np.int64)]
    a, b, c, d = [(words[2 * i] << 32) | words[2 * i + 1] for i in range(4)]
    return a | 1, b, c | 1, d


def ref_hash(name, j, D=D_DEFAULT, seed=0):
    """Exact-integer reference of (bucket, sign) for element j (used by the tests)."""
    a, b, c, d = tensor_coeffs(name, seed)
    shift = 64 - int(np.log2(D))
    h = (((a * j + b) & _MASK64) >> shift)
    s = 1 - 2 * ((((c * j + d) & _MASK64) >> 63) & 1)
    return h, s


def _hash(j, coeffs, D):
    a, b, c, d = (_to_i64(x) for x in coeffs)
    shift = 64 - int(np.log2(D))
    h = ((j * a + b) >> shift) & (D - 1)  # arithmetic shift, then mask = logical shift
    sign = 1.0 - 2.0 * (((j * c + d) >> 63) & 1).to(torch.float32)
    return h, sign


def sketch_add(out, name, values, offset=0, seed=0):
    """out (D,) float32 += sketch of values (flat view of elements offset..offset+n-1 of tensor `name`)."""
    D = out.shape[0]
    assert D & (D - 1) == 0, 'D must be a power of two'
    coeffs = tensor_coeffs(name, seed)
    v = values.reshape(-1)
    n = v.numel()
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        j = torch.arange(offset + s, offset + e, device=v.device, dtype=torch.int64)
        h, sign = _hash(j, coeffs, D)
        out.index_add_(0, h, v[s:e].to(torch.float32) * sign)
    return out


def sketch_named(named_tensors, D=D_DEFAULT, seed=0, device=None):
    """Sketch of a full vector given as (name, tensor) pairs."""
    out = None
    for name, t in named_tensors:
        if t is None:
            continue
        if out is None:
            out = torch.zeros(D, dtype=torch.float32, device=device or t.device)
        sketch_add(out, name, t.detach(), 0, seed)
    return out


def clean_name(name):
    """Canonical parameter name: strip FSDP / checkpoint wrapper prefixes."""
    for tok in ('_fsdp_wrapped_module.', '_checkpoint_wrapped_module.', 'module.'):
        name = name.replace(tok, '')
    return name


def grad_sketch(module, D=D_DEFAULT, seed=0, names=None):
    """Sketch of the .grad of an unwrapped module; names optionally restricts to a parameter subset."""
    out = None
    for n, p in module.named_parameters():
        n = clean_name(n)
        if p.grad is None or (names is not None and n not in names):
            continue
        if out is None:
            out = torch.zeros(D, dtype=torch.float32, device=p.grad.device)
        sketch_add(out, n, p.grad, 0, seed)
    return out


def fsdp_grad_sketch(fsdp_root, D=D_DEFAULT, seed=0, names=None):
    """Sketch of the full (unsharded) gradient of an FSDP model (use_orig_params=False) from each
    rank's local flat-parameter shard, all-reduced. Canonical names match grad_sketch on the
    unwrapped model."""
    import torch.distributed as dist
    out = torch.zeros(D, dtype=torch.float32, device=torch.cuda.current_device())
    for name, start, seg in fsdp_local_grad_segments(fsdp_root, names):
        sketch_add(out, name, seg.to(out.device), start, seed)
    if dist.is_initialized():
        dist.all_reduce(out)
    return out


def fsdp_local_grad_segments(fsdp_root, names=None):
    """Yield (canonical name, start index within the tensor, 1-D grad segment) for every original
    parameter piece held in this rank's flat-parameter shards."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    prefixes = {id(m): n for n, m in fsdp_root.named_modules()}
    for m in FSDP.fsdp_modules(fsdp_root):
        handle = m._handle
        if handle is None:
            continue
        flat = handle.flat_param
        if flat.grad is None:
            continue
        g = flat.grad.reshape(-1)
        prefix = prefixes[id(m)]
        prefix = (prefix + '.') if prefix else ''
        for info, fqn in zip(flat._shard_param_infos, flat._fqns):
            if not info.in_shard:
                continue
            name = clean_name(prefix + fqn)
            if names is not None and name not in names:
                continue
            yield name, info.intra_param_start_idx, g[info.offset_in_shard:info.offset_in_shard + info.numel_in_shard]


def fsdp_full_grads(fsdp_root, names):
    """Full fp32 gradients {canonical name: tensor} of the listed parameters, assembled from the
    local flat-parameter shards and all-reduced (every rank gets the full tensors)."""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    names = set(names)
    shapes = {}
    prefixes = {id(m): n for n, m in fsdp_root.named_modules()}
    for m in FSDP.fsdp_modules(fsdp_root):
        if m._handle is None:
            continue
        flat = m._handle.flat_param
        prefix = prefixes[id(m)]
        prefix = (prefix + '.') if prefix else ''
        for fqn, shape in zip(flat._fqns, flat._shapes):
            n = clean_name(prefix + fqn)
            if n in names:
                shapes[n] = tuple(shape)
    dev = torch.cuda.current_device()
    out = {n: torch.zeros(int(np.prod(s)), dtype=torch.float32, device=dev) for n, s in sorted(shapes.items())}
    for name, start, seg in fsdp_local_grad_segments(fsdp_root, names):
        out[name][start:start + seg.numel()] += seg.to(dev, torch.float32)
    for n in out:
        if dist.is_initialized():
            dist.all_reduce(out[n])
        out[n] = out[n].view(shapes[n]).cpu()
    return out


def rel_l2(x, y):
    return float((x - y).norm() / y.norm().clamp_min(1e-30))
