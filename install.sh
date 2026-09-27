#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Sheba Medical Center. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copy the diploid tasks into a GFMBench-API checkout.
#
# A PYTHONPATH overlay cannot work: gfmbench_api is a regular package, so Python resolves it
# to a single directory rather than merging two.
#
# Usage: ./install.sh /path/to/GFMBench-api
set -euo pipefail

DEST="${1:-}"
if [ -z "$DEST" ]; then
    echo "usage: $0 /path/to/GFMBench-api" >&2
    exit 1
fi
if [ ! -d "$DEST/gfmbench_api/tasks/concrete" ]; then
    echo "error: $DEST does not look like a GFMBench-API checkout" >&2
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
cp -R "$HERE/gfmbench_api" "$HERE/usage_examples" "$DEST/"

echo "Installed into $DEST:"
echo "  gfmbench_api/tasks/concrete/clinvar_diploid_task.py"
echo "  gfmbench_api/tasks/concrete/clinvar_diploid_indel_task.py"
echo "  gfmbench_api/utils/diploidutils.py"
echo "  usage_examples/run_diploid_benchmark.py"
echo "  usage_examples/trainers/linear_probe.py"
echo
echo "Note: usage_examples/trainers/__init__.py was overwritten. It is upstream's file plus"
echo "two exports (SNVEmbeddingProbeFitter, SequenceEmbeddingProbeFitter); GFMFinetuner and"
echo "GFMWithProjection are still re-exported from upstream's own modules."
