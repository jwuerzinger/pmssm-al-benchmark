#!/bin/bash
# Unpack the tutorial's data sample (12 MB, shipped with the repository as
# data/tutorial/pmssm-al-tutorial-data-v1.tar.gz) into data/tutorial/{pool,mcmc}.
#
#   bash scripts/tutorial/fetch_data.sh
#
# Env: TUTORIAL_DATA_URL  where the tarball is (an https URL or a local path;
#                         default: the copy in the repository)
#      TUTORIAL_DATA      where to unpack (default: <repo>/data/tutorial)
#      TUTORIAL_DATA_SHA256  expected checksum ("" skips the check)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${TUTORIAL_DATA:-${REPO_ROOT}/data/tutorial}"
URL="${TUTORIAL_DATA_URL:-${REPO_ROOT}/data/tutorial/pmssm-al-tutorial-data-v1.tar.gz}"
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
