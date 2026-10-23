#!/usr/bin/env python3
"""Decode the timing record a PM_DIAG=1 build writes over the start of c.bin.

    python3 tests/diag_decode.py <c.bin>
"""
import struct
import sys

NAMES = {
    -100: ("setup", ["N", "L", "K", "cpus", "pool workers", "chunks in (a+b)", "chunks out", "bytes/poly"]),
    -101: ("copy engine, one poly (us)", ["H2D pinned", "D2H pinned"]),
    -102: ("host copies, one poly (us)", ["memcpy 1 thread", "pool flat memcpy", "pool stage (gather)", "pool unstage (scatter)"]),
    -103: ("host overheads (us)", ["pool round trip", "event record+wait", "kernel launch"]),
    -104: ("kernels, per launch (us)", ["decompose 1/8", "col fwd 1/8", "row fwd (all)", "row fused (all)",
                                        "col inv 1/16", "crt 1/16", "col inv 1/64", "crt 1/64"]),
    -107: ("register / zero-copy (us)", ["cudaHostRegister 1 poly", "H2D from registered", "cudaHostUnregister",
                                         "kernel write pinned 1 poly", "kernel read pinned 1 poly"]),
    -105: ("whole multiply in setup, host (us)", None),
    -106: ("DMA under host-copy load (us)", ["memcpy x32: DMA", "memcpy x32: copy", "memcpy x6: DMA", "memcpy x6: copy",
                                             "stream x32: DMA", "stream x32: copy", "stream x6: DMA", "stream x6: copy",
                                             "stream gather alone", "stream scatter alone"]),
    -1: ("run: host timeline (us)", None),
    -2: ("run: GPU H2D chunk done (us from start)", None),
    -3: ("run: GPU input chunk compute done", None),
    -4: ("run: GPU fused done", None),
    -5: ("run: GPU CRT chunk done", None),
    -6: ("run: GPU D2H chunk done", None),
}


def main(path):
    raw = open(path, "rb").read()
    magic, n = struct.unpack_from("<II", raw, 0)
    if magic != 0xD1A6D1A6:
        sys.exit("no diagnostic record in %s" % path)
    vals = struct.unpack_from("<%dd" % n, raw, 8)
    i = 0
    while i < n:
        tag, cnt = int(vals[i]), int(vals[i + 1])
        body = vals[i + 2:i + 2 + cnt]
        title, labels = NAMES.get(tag, ("tag %d" % tag, None))
        print("== " + title)
        if labels:
            for lab, v in zip(labels, body):
                print("   %-28s %12.1f" % (lab, v))
        else:
            print("   " + " ".join("%.1f" % v for v in body))
        i += 2 + cnt


if __name__ == "__main__":
    main(sys.argv[1])
