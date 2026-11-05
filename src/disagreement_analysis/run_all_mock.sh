#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# run_all_mock.sh
#
# Clean example batch runner for the two bundled disagreement-analysis
# projects (yelp + ghc-syn). It performs *real* runs — it is not a
# dry-run unless you set CHARDIS_DRY_RUN=1.
#
# Use this as a template for your own batch runners. It deliberately omits
# any environment-specific auth bootstrapping. Authentication is whatever
# your YAML configures:
#   Hosted API → set provider to compatible or openai, auth_mode to api-key,
#                and api_key_env to an exported credential variable such as
#                MODEL_API_KEY. Bind credential_endpoint and credential_provider.
#   Local API  → set provider to local and auth_mode to none on a literal
#                loopback endpoint.
#
# Usage (from repo root):
#   bash src/disagreement_analysis/run_all_mock.sh
#   bash src/disagreement_analysis/run_all_mock.sh yelp
#   VERSIONS="1 2" bash src/disagreement_analysis/run_all_mock.sh
#   CHARDIS_DRY_RUN=1 bash src/disagreement_analysis/run_all_mock.sh
#
# Tunables (env vars):
#   VERSIONS="1"              # space-separated version numbers to run
#   MAX_ATTEMPTS=3            # max attempts per project×version
#   RETRY_BASE_DELAY=60       # base seconds between attempts (exponential backoff)
#   PYTHON=python3            # python interpreter
#   CHARDIS_SKIP_HELDOUT=1    # validation-only run (skip Stage 6)
#   CHARDIS_DRY_RUN=1         # print the planned commands and exit
# ──────────────────────────────────────────────────────────────────────────

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ─── Defaults ────────────────────────────────────────────────────────────

ALL_PROJECTS=(yelp ghc-syn)
if [ "$#" -gt 0 ]; then
    PROJECTS=("$@")
else
    PROJECTS=("${ALL_PROJECTS[@]}")
fi

VERSIONS="${VERSIONS:-1}"
export CHARDIS_SKIP_HELDOUT="${CHARDIS_SKIP_HELDOUT:-0}"

DEFAULT_PYTHON=".venv/bin/python"
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$DEFAULT_PYTHON" ]; then
        PYTHON="$DEFAULT_PYTHON"
    else
        PYTHON="python3"
    fi
fi

MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RETRY_BASE_DELAY="${RETRY_BASE_DELAY:-60}"
DRY_RUN="${CHARDIS_DRY_RUN:-0}"

declare -a SUCCESSES
declare -a FAILURES

# ─── Helpers ─────────────────────────────────────────────────────────────

# Classify a per-project log tail to decide whether to retry.
# Returns 0 if the failure looks transient (worth retrying), 1 if permanent.
is_transient_failure() {
    local log="$1"
    [ -f "$log" ] || return 1
    # Deterministic eval failures — never worth retrying.
    if grep -qiE \
        'FATAL \(non-transient\)|Held-out rename mismatch|needs samples of at least 2 classes' \
        "$log"; then
        return 1
    fi
    # Authentication failures require explicit configuration or credential changes.
    if grep -qiE \
        -e 'CredentialUnavailableError|ClientAuthenticationError|token has expired|ExpiredTokenException|HTTP (401|403)|Unauthorized|Forbidden' \
        -e 'Select provider|requires an explicit supported provider|requires explicit auth_mode|Hosted providers require auth_mode' \
        -e 'Explicitly rebind credential_|Required credential environment variable|Plaintext keys require' \
        "$log"; then
        echo "    Authentication or connection configuration failed." >&2
        echo "    Check provider, auth_mode, the exported credential, and destination bindings." >&2
        return 1
    fi
    # Transient patterns (rate limits, 5xx, network blips).
    if grep -qiE \
        'RateLimitError|429|Too Many Requests|Retry-After|503 |502 |504 |ServiceUnavailable|InternalServerError|ServiceRequestError|getaddrinfo|Temporary failure|Connection (reset|aborted|refused|timed out)|ReadTimeout|RemoteProtocolError|ssl.SSLError|httpx.ConnectError|httpx.ReadError|APIConnectionError|APITimeoutError|gateway timed out' \
        "$log"; then
        return 0
    fi
    return 1
}

# ─── Header ──────────────────────────────────────────────────────────────

START_TS="$(date +%s)"
echo "═══════════════════════════════════════════════════════════════"
echo " Disagreement-eval batch run (mock template)"
echo " Repo root      : $REPO_ROOT"
echo " Projects       : ${PROJECTS[*]}"
echo " Versions       : ${VERSIONS}"
echo " Max attempts   : $MAX_ATTEMPTS  (base backoff ${RETRY_BASE_DELAY}s)"
echo " Skip held-out  : $CHARDIS_SKIP_HELDOUT"
echo " Python         : $PYTHON"
echo " Dry run        : $DRY_RUN"
echo " Started        : $(date)"
echo "═══════════════════════════════════════════════════════════════"

# ─── Main loop ───────────────────────────────────────────────────────────

for project in "${PROJECTS[@]}"; do
  for VERSION in $VERSIONS; do
    cfg="configs/disagreement_configs/${project}/${project}_v${VERSION}.yaml"
    log_dir="outputs/disagreement_outputs/${project}/v${VERSION}"
    log_file="${log_dir}/run_log.txt"

    echo
    echo "───────────────────────────────────────────────────────────────"
    echo "▶ ${project}  v${VERSION}"
    echo "    config : ${cfg}"
    echo "    log    : ${log_file}"
    echo "───────────────────────────────────────────────────────────────"

    if [ ! -f "$cfg" ]; then
        echo "  ✗ Config not found: $cfg — skipping."
        FAILURES+=("${project} v${VERSION} (missing config)")
        continue
    fi

    # Print dataset sizes for this run if we have them.
    val_csv="${log_dir}/validation/validation.csv"
    tst_csv="${log_dir}/heldout/test.csv"
    expl_json="${log_dir}/disagreement_explanations.json"
    val_n="?"; tst_n="?"; expl_n="?"
    [ -f "$val_csv" ] && val_n=$("$PYTHON" -c "import pandas; print(len(pandas.read_csv('$val_csv')))" 2>/dev/null || echo "?")
    [ -f "$tst_csv" ] && tst_n=$("$PYTHON" -c "import pandas; print(len(pandas.read_csv('$tst_csv')))" 2>/dev/null || echo "?")
    [ -f "$expl_json" ] && expl_n=$("$PYTHON" -c "import json; print(len(json.load(open('$expl_json', encoding='utf-8-sig'))))" 2>/dev/null || echo "?")
    echo "  · Hypotheses: ${expl_n}    Validation: ${val_n} rows    Test: ${tst_n} rows"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  (dry run) would invoke:"
        echo "    $PYTHON -m src.run_disagreement_cli --skip-setup --project $project --version $VERSION"
        SUCCESSES+=("${project} v${VERSION} (dry run)")
        continue
    fi

    mkdir -p "$log_dir"
    proj_start="$(date +%s)"
    rc=1
    attempt=1

    while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
        echo "  · Attempt ${attempt}/${MAX_ATTEMPTS}  ($(date))"

        {
            echo "================================================================"
            echo "Attempt ${attempt}/${MAX_ATTEMPTS} at $(date)"
            echo "================================================================"
        } >> "$log_file"

        "$PYTHON" -m src.run_disagreement_cli \
            --skip-setup \
            --project "$project" \
            --version "$VERSION" \
            2>&1 | tee -a "$log_file"
        rc=${PIPESTATUS[0]}

        if [ "$rc" -eq 0 ]; then
            break
        fi

        if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
            echo "  ✗ Attempt ${attempt} failed (exit ${rc}) — no attempts remaining."
            break
        fi

        if is_transient_failure "$log_file"; then
            delay=$(( RETRY_BASE_DELAY * (1 << (attempt - 1)) ))
            echo "  ⟳ Transient failure detected (exit ${rc}). Sleeping ${delay}s before retry…"
            sleep "$delay"
            attempt=$(( attempt + 1 ))
            continue
        fi

        echo "  ✗ Failure does not look transient (exit ${rc}); not retrying."
        break
    done

    proj_elapsed=$(( $(date +%s) - proj_start ))
    if [ "$rc" -eq 0 ]; then
        echo "  ✓ ${project} v${VERSION} completed in ${proj_elapsed}s after ${attempt} attempt(s)"
        SUCCESSES+=("${project} v${VERSION} (${proj_elapsed}s, ${attempt} attempt(s))")
    else
        echo "  ✗ ${project} v${VERSION} FAILED (exit ${rc}) after ${proj_elapsed}s — see ${log_file}"
        FAILURES+=("${project} v${VERSION} (exit ${rc} after ${attempt} attempt(s))")
    fi
  done
done

# ─── Summary ─────────────────────────────────────────────────────────────

TOTAL_ELAPSED=$(( $(date +%s) - START_TS ))
echo
echo "═══════════════════════════════════════════════════════════════"
echo " Batch complete in ${TOTAL_ELAPSED}s"
echo "   Successes (${#SUCCESSES[@]}):"
for s in "${SUCCESSES[@]:-}"; do [ -n "$s" ] && echo "     • $s"; done
echo "   Failures  (${#FAILURES[@]}):"
for f in "${FAILURES[@]:-}"; do [ -n "$f" ] && echo "     • $f"; done
echo "═══════════════════════════════════════════════════════════════"
echo
echo " Tip: re-run with the failed project names as positional args to retry"
echo " just those (eval skip-guards mean completed stages won't repeat)."

[ "${#FAILURES[@]}" -eq 0 ] || exit 1
exit 0
