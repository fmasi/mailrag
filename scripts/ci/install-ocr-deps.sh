#!/usr/bin/env bash
# Install the OCR binaries the test suite needs (tesseract, poppler) on an
# Ubuntu CI runner, retrying when the package mirrors are unreachable.
#
# On 11 Nov 2026 the runner could resolve none of its apt mirrors (the
# runner's mirror list names azure.archive.ubuntu.com, archive.ubuntu.com and
# security.ubuntu.com, and all three failed), and the required `pytest` check
# died here three times, about 10 minutes each, before a single test ran.
# Rewriting the apt sources to another mirror would not help: the mirror list
# already falls back. What does help is retrying a short blip, and failing in
# a few minutes rather than ten when the outage is real, so a later rerun is
# cheap.
set -euo pipefail

PACKAGES=(tesseract-ocr poppler-utils)
APT_OPTS=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
BACKOFF=(30 90) # seconds to wait before attempts 2 and 3

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# `--error-on=any` makes `apt-get update` exit non-zero when an index fails to
# download; by default it only warns, and the install then fails confusingly.
# Only the network steps are time-boxed: killing dpkg mid-install could leave
# it interrupted for the next attempt.
apt_install() {
  $SUDO timeout 90 apt-get "${APT_OPTS[@]}" update --error-on=any &&
    $SUDO timeout 180 apt-get "${APT_OPTS[@]}" install -y --download-only "${PACKAGES[@]}" &&
    $SUDO apt-get install -y "${PACKAGES[@]}"
}

for wait in "${BACKOFF[@]}"; do
  if apt_install; then
    exit 0
  fi
  echo "::warning::apt attempt failed (see the output above), retrying in ${wait}s"
  sleep "$wait"
done
apt_install || {
  echo "::error::apt failed 3 times; if the mirrors were unreachable, rerun once they recover"
  exit 1
}
