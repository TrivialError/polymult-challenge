#!/usr/bin/env bash
# One measurement round on FHERMA, for the maintainer's own implementation.
#
#   tests/fherma_round.sh diag  [seeds]   PM_DIAG=1 build from the `diag` branch (fails verification by design)
#   tests/fherma_round.sh main  [seeds]   the current `main` commit, both challenge points, verified
#
# Rebuilds `diag` as main + PM_DIAG=1 when asked, points the implementation's
# repository artifact at the commit, queues a run (waiting out the platform's
# cooldown) and prints the result. Needs the fherma CLI logged in.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:?diag or main}
seeds=${2:-12}
FPY="${FHERMA_PY:-$HOME/Library/Application Support/pipx/venvs/fherma/bin/python}"

git push -q origin main
if [[ "$mode" == diag ]]; then
  git checkout -q diag
  git reset -q --hard main
  sed -i '' 's/^#define PM_DIAG 0$/#define PM_DIAG 1/' solve.cu
  git commit -qam "Diagnostic build (PM_DIAG=1) - not for submission"
  git push -q -f origin diag
  commit=$(git rev-parse HEAD)
  git checkout -q main
  body="{\"points\": [{\"point\": {\"N\": 65536, \"W\": 1747}}], \"seeds\": {\"count\": $seeds}}"
else
  commit=$(git rev-parse HEAD)
  body="{\"seeds\": {\"count\": $seeds}}"
fi
echo "commit $commit"

"$FPY" - "$commit" "$body" <<'PY'
import json, re, sys, time
from fherma import config
from fherma.api import Api

api = Api(config.load())
base = "/kernels/polynomial-multiplication/implementations/rns-ntt-pipeline"
commit, body = sys.argv[1], json.loads(sys.argv[2])
impl = api._request("GET", base)
art = impl["artifacts"]
art[0]["digest"] = commit
api._request("PATCH", base, json={"artifacts": art})
while True:
    try:
        print("queued", api._request("POST", base + "/benchmark", json=body)["id"], flush=True)
        break
    except Exception as e:
        hint = str(getattr(e, "hint", ""))
        m = re.search(r"in (\d+) (minute|second)", hint)
        if not m:
            raise
        wait = int(m.group(1)) * (60 if m.group(2) == "minute" else 1) + 15
        print("cooldown %ds" % wait, flush=True)
        time.sleep(min(wait, 120))
while True:
    runs = [r for r in api._request("GET", base + "/runs")["items"] if r["commit"] == commit]
    if runs and runs[0]["status"] in ("finished", "failed", "cancelled", "lost"):
        r = runs[0]
        print("run", r["id"], r["status"], r.get("verdict"), "%s/%s" % (r.get("passed"), r.get("cases")), r.get("failure") or "")
        for p in r.get("points") or []:
            print("  %-14s median %s  %s/%s" % (p["point"], p.get("median_s"), p.get("passed"), p.get("seeds")))
        break
    time.sleep(20)
PY
