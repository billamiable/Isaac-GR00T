#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="${TRAINING_ENV_SH:-${SCRIPT_DIR}/training_paths.sh}"

if [[ -f "${ENV_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SH}"
fi

usage() {
  cat <<'EOF'
Verify and optionally unpack GenieSim task archives.

Usage:
  verify_and_unpack_geniesim_archives.sh [instruction_root] [--extract]

Examples:
  # Use RAW_ROOT from training_paths.sh
  bash scripts/preprocess_agibot/verify_and_unpack_geniesim_archives.sh --extract

  # Verify only (no file writes)
  bash scripts/preprocess_agibot/verify_and_unpack_geniesim_archives.sh \
    /ephemeral/agibot/instruction

  # Verify and extract
  bash scripts/preprocess_agibot/verify_and_unpack_geniesim_archives.sh \
    /ephemeral/agibot/instruction --extract
EOF
}

if [[ $# -ge 1 && "${1}" != "--extract" ]]; then
  INSTRUCTION_ROOT="$1"
  shift
else
  INSTRUCTION_ROOT="${RAW_ROOT:-}"
fi

DO_EXTRACT=0
if [[ "${1:-}" == "--extract" ]]; then
  DO_EXTRACT=1
  shift
fi

if [[ $# -ne 0 ]]; then
  usage
  exit 1
fi

if [[ -z "${INSTRUCTION_ROOT}" ]]; then
  usage
  exit 1
fi

if [[ ! -d "${INSTRUCTION_ROOT}" ]]; then
  echo "Error: instruction root does not exist: ${INSTRUCTION_ROOT}" >&2
  exit 1
fi

extract_series() {
  local dir="$1"
  local base_name="$2"

  local split_head="${dir}/${base_name}.000"
  local plain_tar="${dir}/${base_name}"

  if compgen -G "${dir}/${base_name}.[0-9][0-9][0-9]" >/dev/null; then
    mapfile -t parts < <(ls "${dir}/${base_name}."[0-9][0-9][0-9] | sort)
    if [[ ${#parts[@]} -eq 1 ]]; then
      echo "  - ${base_name}: single .000 file (treat as full tar.gz)"
      if [[ ${DO_EXTRACT} -eq 1 ]]; then
        tar -xzf "${parts[0]}" -C "${dir}"
      fi
    else
      echo "  - ${base_name}: split archive with ${#parts[@]} part(s)"
      if [[ ${DO_EXTRACT} -eq 1 ]]; then
        cat "${parts[@]}" | tar -xzf - -C "${dir}"
      fi
    fi
    return 0
  fi

  if [[ -f "${plain_tar}" ]]; then
    echo "  - ${base_name}: plain tar.gz file"
    if [[ ${DO_EXTRACT} -eq 1 ]]; then
      tar -xzf "${plain_tar}" -C "${dir}"
    fi
    return 0
  fi

  echo "  - ${base_name}: NOT FOUND"
  return 1
}

echo "Instruction root: ${INSTRUCTION_ROOT}"
echo "Mode: $([[ ${DO_EXTRACT} -eq 1 ]] && echo "verify + extract" || echo "verify only")"
echo

ok_count=0
fail_count=0
task_count=0

for task_dir in "${INSTRUCTION_ROOT}"/*; do
  [[ -d "${task_dir}" ]] || continue
  task_count=$((task_count + 1))
  echo "[Task] $(basename "${task_dir}")"

  task_ok=1
  extract_series "${task_dir}" "meta.tar.gz" || task_ok=0
  extract_series "${task_dir}" "data.tar.gz" || task_ok=0
  extract_series "${task_dir}" "videos.tar.gz" || task_ok=0

  if [[ ${DO_EXTRACT} -eq 1 ]]; then
    if [[ -f "${task_dir}/meta/info.json" && -d "${task_dir}/data" && -d "${task_dir}/videos" ]]; then
      echo "  - extracted structure looks good (meta/info.json + data + videos)"
    else
      echo "  - extracted structure is incomplete"
      task_ok=0
    fi
  fi

  if [[ ${task_ok} -eq 1 ]]; then
    ok_count=$((ok_count + 1))
    echo "  -> status: OK"
  else
    fail_count=$((fail_count + 1))
    echo "  -> status: FAIL"
  fi
  echo
done

echo "Summary: tasks=${task_count}, ok=${ok_count}, fail=${fail_count}"
if [[ ${fail_count} -gt 0 ]]; then
  exit 2
fi

