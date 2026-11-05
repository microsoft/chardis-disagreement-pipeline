#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [ "${1:-}" = "--setup" ]; then
    shift
    if [ ! -x .venv/bin/python ]; then
        python3.11 -m venv .venv
    fi
    LOCK="$(.venv/bin/python -m src.environment --lock)"
    .venv/bin/python -m pip install --require-hashes -r "$LOCK"
    for option in "$@"; do
        case "$option" in
            --browser)
                PROFILE="${option#--}"
                LOCK="$(.venv/bin/python -m src.environment --lock --profile "$PROFILE")"
                .venv/bin/python -m pip install --require-hashes -r "$LOCK"
                if [ "$PROFILE" = "browser" ]; then
                    .venv/bin/python -m playwright install chromium
                fi
                ;;
            *)
                echo "Unknown setup option. Use --browser." >&2
                exit 1
                ;;
        esac
    done
    .venv/bin/python -m src.environment
    echo "Setup complete. Launch the pipeline separately."
    exit 0
fi
if [ ! -x .venv/bin/python ]; then
    echo "Run bash src/run_disagreement.sh --setup first." >&2
    exit 1
fi
.venv/bin/python -m src.environment
exec .venv/bin/python -m src.run_disagreement_cli "$@"
