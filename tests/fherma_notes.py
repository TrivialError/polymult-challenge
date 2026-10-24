#!/usr/bin/env python3
"""Reassemble a PM_DIAG timing record from the crash notes of a FHERMA run and
print it like diag_decode.py. Run with the fherma CLI's Python, which carries
the API client and the stored login:

    "<fherma venv>/bin/python" tests/fherma_notes.py <run id>
"""
import os
import re
import sys

from fherma import config
from fherma.api import Api

sys.path.insert(0, os.path.dirname(__file__))
from diag_decode import NAMES  # noqa: E402


def main(run_id):
    api = Api(config.load())
    run = api._request("GET", "/runs/%s" % run_id)["run"]
    slices, total = {}, 0
    print("per call (m = plain memcpy staging): take, stage0, H2D first, H2D last, input computed, fused, "
          "CRT first, D2H first, D2H last, host submit done, host end, host in submits, host waiting staging (us)")
    for p in run["points"]:
        for c in p["cases"]:
            note = c.get("note") or ""
            r = re.match(r"R(\d+[km]?) ([^|]*)", note)
            if r:
                body = r.group(2).strip()
                if " -9 " in " " + body + " ":
                    head, _, per = body.partition("-9")
                    v = [float(x) for x in per.split()]
                    print("   call %-4s %s" % (r.group(1), head.strip()))
                    print("        chunk issue->done (us):", " ".join("%d>%d" % (v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)))
                else:
                    print("   call %-4s %s" % (r.group(1), body))
            m = re.search(r"D(\d+)/(\d+) (.*)", note, re.S)
            if m:
                slices[int(m.group(1))] = m.group(3)
                total = int(m.group(2))
    missing = [k for k in range(total) if k not in slices]
    if missing:
        print("missing slices", missing, "- values around them are unreliable")
    text = "".join(slices.get(k, "") for k in range(total))
    vals = [float(t) for t in text.split()]
    i = 0
    while i + 1 < len(vals):
        tag, cnt = int(vals[i]), int(vals[i + 1])
        if tag >= 0 or cnt < 0:
            i += 1
            continue
        body = vals[i + 2:i + 2 + cnt]
        title, labels = NAMES.get(tag, ("tag %d" % tag, None))
        print("== " + title)
        if labels:
            for lab, v in zip(labels, body):
                print("   %-28s %12.1f" % (lab, v))
        else:
            print("   " + " ".join("%.0f" % v for v in body))
        i += 2 + cnt


if __name__ == "__main__":
    main(sys.argv[1])
