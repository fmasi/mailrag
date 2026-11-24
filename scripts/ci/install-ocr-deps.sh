#!/usr/bin/env bash
# Install the OCR binaries the test suite needs (tesseract, poppler) on an
# Ubuntu CI runner, surviving a flaky package mirror.
#
# GitHub's Ubuntu runners fetch from azure.archive.ubuntu.com. In November 2026
# that mirror stopped resolving for over an hour and failed the required
# `pytest` check three times before a single test ran. So: try the configured
# mirror twice, then rewrite the apt sources to archive.ubuntu.com and try once
# more. Each apt call is time-boxed so a hung mirror fails fast instead of
# eating the job's timeout.
set -euo pipefail

PACKAGES=(tesseract-ocr poppler-utils)
FALLBACK_MIRROR="http://archive.ubuntu.com/ubuntu"
APT_OPTS=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30)

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# `--error-on=any` makes `apt-get update` exit non-zero when an index fails to
# download; by default it only warns, and the install then fails confusingly.
apt_install() {
  $SUDO timeout 90 apt-get "${APT_OPTS[@]}" update --error-on=any &&
    $SUDO timeout 180 apt-get "${APT_OPTS[@]}" install -y "${PACKAGES[@]}"
}

for attempt in 1 2; do
  if apt_install; then
    exit 0
  fi
  echo "::warning::apt attempt ${attempt} against the configured mirror failed"
  sleep 15
done

echo "::warning::switching apt sources to ${FALLBACK_MIRROR}"
for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list /etc/apt/sources.list.d/*.sources; do
  [ -f "$f" ] || continue
  $SUDO sed -i -E "s#https?://[a-z0-9.-]*archive\.ubuntu\.com/ubuntu/?#${FALLBACK_MIRROR}/#g" "$f"
done
apt_install
