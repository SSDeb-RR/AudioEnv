#!/bin/sh
set -eu

test -f /workspace/seed.txt
test -f /workspace/fixed.txt
test ! -f /workspace/wrong.txt
