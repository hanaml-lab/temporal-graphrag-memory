#!/usr/bin/env bash
# Regenerate every table and figure of the report from its seeded configuration.
set -euo pipefail
cd "$(dirname "$0")/paper"
python make_data.py
python figures.py
echo "tables in scripts/paper/tables, figures in scripts/paper/figures"
