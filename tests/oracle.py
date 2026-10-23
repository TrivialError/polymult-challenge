#!/usr/bin/env python3
"""Independent generator + verifier for local testing (not the platform bundle).

Writes point directories in the layout main.cpp reads and computes the expected
product with Python big integers: a negacyclic NTT directly modulo q (fast enough
for N = 65536, W = 1747), cross-checked against schoolbook multiplication on
small sizes by `selftest`.

    python3 tests/oracle.py make   <dir> <N> <W> [cases] [mode]
    python3 tests/oracle.py verify <dir>
    python3 tests/oracle.py selftest

mode: random (default) | max (all q-1) | zero | mixed (cases cycle through
random, max, sparse, small)
"""
import json
import os
import random
import struct
import sys


def is_prime(n, rounds=32):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    rng = random.Random(0xC0FFEE)
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def negacyclic_prime(N, W):
    """The largest prime below 2^W with q == 1 (mod 2N), as the spec derives q."""
    m = 2 * N
    q = (1 << W) - 1
    q -= (q - 1) % m
    while q > 1:
        if is_prime(q):
            return q
        q -= m
    raise RuntimeError("no prime found")


def psi_root(q, N):
    """A primitive 2N-th root of unity mod q."""
    e = (q - 1) // (2 * N)
    for g in range(2, 10000):
        x = pow(g, e, q)
        if pow(x, N, q) == q - 1:
            return x
    raise RuntimeError("no root")


def bitrev(k, bits):
    r = 0
    for _ in range(bits):
        r = (r << 1) | (k & 1)
        k >>= 1
    return r


def negacyclic_ntt_mul(a, b, q, N):
    if N == 1:
        return [a[0] * b[0] % q]
    logn = N.bit_length() - 1
    psi = psi_root(q, N)
    ipsi = pow(psi, q - 2, q)
    pw = [1] * N
    ipw = [1] * N
    for i in range(1, N):
        pw[i] = pw[i - 1] * psi % q
        ipw[i] = ipw[i - 1] * ipsi % q
    tw = [pw[bitrev(k, logn)] for k in range(N)]
    itw = [ipw[bitrev(k, logn)] for k in range(N)]

    def fwd(x):
        x = list(x)
        t, m = N, 1
        while m < N:
            t >>= 1
            for i in range(m):
                j1 = 2 * i * t
                S = tw[m + i]
                for j in range(j1, j1 + t):
                    U = x[j]
                    V = x[j + t] * S % q
                    x[j] = (U + V) % q
                    x[j + t] = (U - V) % q
            m <<= 1
        return x

    def inv(x):
        x = list(x)
        t, m = 1, N
        while m > 1:
            h = m >> 1
            j1 = 0
            for i in range(h):
                S = itw[h + i]
                for j in range(j1, j1 + t):
                    U = x[j]
                    V = x[j + t]
                    x[j] = (U + V) % q
                    x[j + t] = (U - V) * S % q
                j1 += 2 * t
            t <<= 1
            m = h
        ninv = pow(N, q - 2, q)
        return [v * ninv % q for v in x]

    A, B = fwd(a), fwd(b)
    return inv([u * v % q for u, v in zip(A, B)])


def negacyclic_schoolbook(a, b, q, N):
    c = [0] * N
    for i in range(N):
        for j in range(N):
            t = i + j
            if t < N:
                c[t] += a[i] * b[j]
            else:
                c[t - N] -= a[i] * b[j]
    return [v % q for v in c]


def pack(values, L):
    out = bytearray()
    mask = (1 << 32) - 1
    for x in values:
        out += struct.pack("<%dI" % L, *[(x >> (32 * i)) & mask for i in range(L)])
    return bytes(out)


def unpack(raw, N, L):
    words = struct.unpack("<%dI" % (N * L), raw)
    return [sum(w << (32 * i) for i, w in enumerate(words[k * L:(k + 1) * L])) for k in range(N)]


def operands(mode, rng, q, N):
    if mode == "max":
        return [q - 1] * N, [q - 1] * N
    if mode == "zero":
        return [0] * N, [0] * N
    if mode == "sparse":
        a, b = [0] * N, [0] * N
        for _ in range(3):
            a[rng.randrange(N)] = rng.randrange(q)
            b[rng.randrange(N)] = rng.randrange(q)
        return a, b
    if mode == "small":
        return [rng.randrange(3) for _ in range(N)], [q - 1 - rng.randrange(3) for _ in range(N)]
    return [rng.randrange(q) for _ in range(N)], [rng.randrange(q) for _ in range(N)]


def make(root, N, W, cases, mode):
    L = (W + 31) // 32
    q = negacyclic_prime(N, W)
    os.makedirs(os.path.join(root, "point"), exist_ok=True)
    with open(os.path.join(root, "point", "q.bin"), "wb") as f:
        f.write(pack([q], L))
    cycle = ["random", "max", "sparse", "small"]
    for s in range(cases):
        rng = random.Random(1000 + 7919 * s + N + W)
        m = cycle[s % len(cycle)] if mode == "mixed" else mode
        a, b = operands(m, rng, q, N)
        case = os.path.join(root, "cases", "%06d" % s)
        exp = os.path.join(root, "expected", "%06d" % s)
        os.makedirs(case, exist_ok=True)
        os.makedirs(exp, exist_ok=True)
        with open(os.path.join(case, "a.bin"), "wb") as f:
            f.write(pack(a, L))
        with open(os.path.join(case, "b.bin"), "wb") as f:
            f.write(pack(b, L))
        with open(os.path.join(exp, "c.bin"), "wb") as f:
            f.write(pack(negacyclic_ntt_mul(a, b, q, N), L))
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"point": {"N": N, "W": W, "L": L}, "N": N, "W": W, "L": L, "cases": cases}, f)
    print("made N=%d W=%d L=%d delta=2^W-q=%d cases=%d mode=%s at %s" % (N, W, L, (1 << W) - q, cases, mode, root))


def verify(root):
    with open(os.path.join(root, "manifest.json")) as f:
        m = json.load(f)
    N, L, cases = m["N"], m["L"], m["cases"]
    ok = True
    for s in range(cases):
        got_path = os.path.join(root, "out", "%06d" % s, "c.bin")
        exp_path = os.path.join(root, "expected", "%06d" % s, "c.bin")
        if not os.path.exists(got_path):
            print("case %06d: no output" % s)
            ok = False
            continue
        got, exp = open(got_path, "rb").read(), open(exp_path, "rb").read()
        if got == exp:
            continue
        ok = False
        if len(got) != len(exp):
            print("case %06d: wrong length %d, want %d" % (s, len(got), len(exp)))
            continue
        g, e = unpack(got, N, L), unpack(exp, N, L)
        diffs = [i for i in range(N) if g[i] != e[i]]
        print("case %06d: MISMATCH in %d/%d coefficients, first at %s" % (s, len(diffs), N, diffs[:8]))
    with open(os.path.join(root, "out", "results.json")) as f:
        res = json.load(f)
    times = sorted(c["seconds"] for c in res["cases"] if c.get("seconds") is not None)
    bad = [c for c in res["cases"] if c.get("status") != "ok"]
    for c in bad:
        print("case %06d: %s %s" % (c["i"], c.get("status"), c.get("note", "")))
    med = times[len(times) // 2] if times else float("nan")
    print("VERIFY %s: %s  (%d cases, median %.3f ms)" % (root, "PASS" if ok and not bad else "FAIL", cases, med * 1e3))
    return ok and not bad


def selftest():
    rng = random.Random(5)
    for N, W in [(1, 8), (2, 16), (4, 20), (8, 33), (16, 64), (32, 100), (64, 32), (128, 256)]:
        q = negacyclic_prime(N, W)
        for _ in range(3):
            a = [rng.randrange(q) for _ in range(N)]
            b = [rng.randrange(q) for _ in range(N)]
            assert negacyclic_ntt_mul(a, b, q, N) == negacyclic_schoolbook(a, b, q, N), (N, W)
    print("selftest: NTT oracle agrees with schoolbook")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "make":
        cases = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        mode = sys.argv[6] if len(sys.argv) > 6 else "random"
        make(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), cases, mode)
    elif len(sys.argv) == 3 and sys.argv[1] == "verify":
        sys.exit(0 if verify(sys.argv[2]) else 1)
    elif len(sys.argv) == 2 and sys.argv[1] == "selftest":
        selftest()
    else:
        print(__doc__)
        sys.exit(2)
