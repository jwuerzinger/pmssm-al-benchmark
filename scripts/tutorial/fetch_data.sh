#!/bin/bash
# Fetch the tutorial's data sample (about 12 MB) into data/tutorial.
#
#   bash scripts/tutorial/fetch_data.sh
#
# Env: TUTORIAL_DATA_URL  where the tarball is (an https URL or a local path;
#                         default: the GitHub release asset)
#      TUTORIAL_DATA      where to unpack (default: <repo>/data/tutorial)
#      TUTORIAL_DATA_SHA256  expected checksum ("" skips the check)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${TUTORIAL_DATA:-${REPO_ROOT}/data/tutorial}"
URL="${TUTORIAL_DATA_URL:-https://github.com/jwuerzinger/pmssm-al-benchmark/releases/download/tutorial-data-v1/pmssm-al-tutorial-data-v1.tar.gz}"
SHA="${TUTORIAL_DATA_SHA256-f4a374a0bc8f837e1347cca0a0ccf5808cfb66224761c9f49a4dc182b8314b20}"

if [[ -f "${DEST}/pool/ntuple.tutorial.0.root" && -f "${DEST}/mcmc/ntuple.tutorial.0.root" ]]; then
    echo "[tutorial] data already present in ${DEST}"
    exit 0
fi
TMP="$(mktemp --suffix=.tar.gz)"
trap 'rm -f "${TMP}"' EXIT
if [[ "${URL}" == /* ]]; then
    cp "${URL}" "${TMP}"
else
    echo "[tutorial] downloading ${URL}"
    curl -fL --progress-bar "${URL}" -o "${TMP}"
fi
if [[ -n "${SHA}" ]]; then
    echo "${SHA}  ${TMP}" | sha256sum -c - >/dev/null || { echo "[tutorial] checksum mismatch; refusing to unpack" >&2; exit 1; }
fi
mkdir -p "${DEST}"
tar -xzf "${TMP}" -C "${DEST}" --strip-components=1
echo "[tutorial] unpacked into ${DEST}:"
ls -l "${DEST}/pool" "${DEST}/mcmc"
