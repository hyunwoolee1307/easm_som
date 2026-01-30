#!/usr/bin/env bash
set -euo pipefail

# Limit CPU threads to ~80% of cores
CORES=$(python - <<'PY'
import os
c = os.cpu_count() or 1
print(max(1, int(c * 0.8)))
PY
)

export OMP_NUM_THREADS="$CORES"
export OPENBLAS_NUM_THREADS="$CORES"
export MKL_NUM_THREADS="$CORES"
export NUMEXPR_NUM_THREADS="$CORES"
export VECLIB_MAXIMUM_THREADS="$CORES"
export BLIS_NUM_THREADS="$CORES"

# Limit Dask memory to ~80% of total RAM
MEM_BYTES=$(python - <<'PY'
import re
mem_kb = 0
with open('/proc/meminfo', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('MemTotal:'):
            mem_kb = int(re.findall(r'\d+', line)[0])
            break
print(int(mem_kb * 1024 * 0.8))
PY
)
export DASK_DISTRIBUTED__WORKER__MEMORY_LIMIT="$MEM_BYTES"

# Run command
exec "$@"
