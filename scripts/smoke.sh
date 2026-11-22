#!/usr/bin/env bash
# End-to-end HTTP smoke test against a running server.
# Usage: scripts/smoke.sh [base_url]   (default http://127.0.0.1:8000)
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== health =="
curl -fsS "$BASE/api/health"; echo

echo "== demo corpus endpoint =="
curl -fsS "$BASE/api/demo-corpus" -o "$TMP/demo.csv"
wc -l "$TMP/demo.csv"

echo "== create session =="
SID=$(curl -fsS -F "file=@$TMP/demo.csv;type=text/csv" "$BASE/api/sessions" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["session_id"])')
echo "session=$SID"

echo "== analyze (satisfied expected) =="
curl -fsS -X POST "$BASE/api/analyze" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 0.85, \"max_fdr\": 0.2
}" -o "$TMP/a1.json"
python3 - "$TMP/a1.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["status"] == "satisfied", d["status"]
print("status:", d["status"], "| expanded:", d["expanded_terms"],
      "| recall:", round(d["metrics"]["recall"], 3),
      "| fdr:", round(d["metrics"]["fdr"], 3))
print("suggestion_id:", d["suggestion_id"], "| rev:", d["revision"])
PY
SUG=$(python3 -c 'import json; print(json.load(open("'$TMP'/a1.json"))["suggestion_id"])')
TERM=$(python3 -c 'import json; print(json.load(open("'$TMP'/a1.json"))["expanded_terms"][0])')

echo "== revise: exclude $TERM (revision bumps) =="
curl -fsS -X POST "$BASE/api/revise" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"exclude\": [\"$TERM\"]
}" -o "$TMP/rev.json"
python3 - "$TMP/rev.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["revision"] == 1 and d["previous_suggestion_invalidated"]
print("rev:", d["revision"], "| excluded:", d["excluded_terms"])
PY

echo "== stale suggestion must return 409 =="
CODE=$(curl -s -o "$TMP/stale.json" -w "%{http_code}" -X POST "$BASE/api/analyze" \
  -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 0.85, \"max_fdr\": 0.2, \"suggestion_id\": \"$SUG\"}")
echo "HTTP $CODE"; test "$CODE" = "409"
python3 -c 'import json; print("detail:", json.load(open("'$TMP'/stale.json"))["detail"])'

echo "== fresh analyze after revision =="
curl -fsS -X POST "$BASE/api/analyze" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 0.85, \"max_fdr\": 0.2
}" -o "$TMP/a2.json"
python3 - "$TMP/a2.json" "$TERM" <<'PY'
import json, sys
d, term = json.load(open(sys.argv[1])), sys.argv[2]
assert term not in d["expanded_terms"], d["expanded_terms"]
print("recomputed expanded:", d["expanded_terms"],
      "| new suggestion:", d["suggestion_id"][:8], "| rev:", d["revision"])
PY

echo "== conflict threshold (failed + diagnosis) =="
curl -fsS -X POST "$BASE/api/analyze" -H "Content-Type: application/json" -d "{
  \"session_id\": \"$SID\", \"query\": \"quantum error correction\",
  \"min_recall\": 1.0, \"max_fdr\": 0.05
}" -o "$TMP/a3.json"
python3 - "$TMP/a3.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["status"] == "failed" and d["failure"]["reasons"]
print("failed as expected; ceiling recall:",
      d["failure"]["unconstrained_max_recall"],
      "| reasons:", len(d["failure"]["reasons"]))
PY

echo "ALL SMOKE CHECKS PASSED"
