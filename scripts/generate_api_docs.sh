#!/usr/bin/env bash
# Generate the developer API documentation of goldener into docs/api.
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="docs/api"
REPO_URL="https://github.com/goldener-data/goldener"

# goldener/__init__.py defines __all__, which hides its submodules from pdoc:
# its direct submodules and subpackages must then be listed explicitly.
MODULES=$(find goldener -mindepth 1 -maxdepth 1 -name "*.py" ! -name "__init__.py" | sed 's#/#.#g; s#\.py$##' | sort)
PACKAGES=$(find goldener -mindepth 2 -maxdepth 2 -name "__init__.py" | xargs -n1 dirname | sed 's#/#.#g' | sort)

rm -rf "$OUTPUT_DIR"
uv run --no-sync pdoc \
    --docformat google \
    --template-directory scripts/api_docs_templates \
    --no-show-source \
    --edit-url "goldener=${REPO_URL}/blob/main/goldener/" \
    --logo-link "${REPO_URL}" \
    --footer-text "goldener $(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)" \
    -o "$OUTPUT_DIR" \
    goldener $PACKAGES $MODULES
