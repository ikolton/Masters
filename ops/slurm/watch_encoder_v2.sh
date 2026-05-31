#!/usr/bin/env bash
# Watcher + auto-recovery for encoder v2 training (j1+j2).
# Safe to call repeatedly; reads/writes state to STATE_FILE.
# Exit codes: 0=ok/nothing-to-do, 2=resubmitted, 3=needs-human

MASTERS=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
OUTPUT_DIR=${MASTERS}/outputs/encoder/v2_15mm_k4_4gpu
STATE_FILE=${MASTERS}/logs/enc_v2_watcher_state.json
LOG=${MASTERS}/logs/enc_v2_watcher.log
MAX_RETRIES=4
PYBIN=/net/scratch/hscra/plgrid/plgikolton/conda-envs/codex-masters-py311/bin/python

mkdir -p "${MASTERS}/logs"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$(ts) $*" | tee -a "${LOG}"; }

# ── state helpers ────────────────────────────────────────────────────────────
read_state() {
    if [[ ! -f "${STATE_FILE}" ]]; then
        # First run — bootstrap from initial job IDs
        ${PYBIN} -c "
import json, pathlib
pathlib.Path('${STATE_FILE}').write_text(json.dumps({
    'j1': '18252104', 'j2': '18252105',
    'retries': 0, 'done': False, 'phase': 'j1'
}, indent=2))
"
    fi
    J1=$(${PYBIN}  -c "import json; print(json.load(open('${STATE_FILE}'))['j1'])")
    J2=$(${PYBIN}  -c "import json; print(json.load(open('${STATE_FILE}'))['j2'])")
    RETRIES=$(${PYBIN} -c "import json; print(json.load(open('${STATE_FILE}'))['retries'])")
    DONE=$(${PYBIN}    -c "import json; print(json.load(open('${STATE_FILE}'))['done'])")
    PHASE=$(${PYBIN}   -c "import json; print(json.load(open('${STATE_FILE}'))['phase'])")
}

write_state() {
    ${PYBIN} -c "
import json, pathlib
pathlib.Path('${STATE_FILE}').write_text(json.dumps({
    'j1': '${J1}', 'j2': '${J2}',
    'retries': ${RETRIES}, 'done': ${DONE_PY}, 'phase': '${PHASE}'
}, indent=2))
"
}

# ── SLURM helpers ────────────────────────────────────────────────────────────
job_live_state() {   # RUNNING / PENDING / "" (not in queue)
    squeue -j "$1" -h -o "%T" 2>/dev/null || true
}

job_final_state() {  # COMPLETED / FAILED / CANCELLED / TIMEOUT / NODE_FAIL / ""
    sacct -j "$1" --format=State --noheader -P 2>/dev/null \
        | head -1 | cut -d'|' -f1 | tr -d ' ' || true
}

job_state() {
    local live; live=$(job_live_state "$1")
    if [[ -n "$live" ]]; then echo "$live"; return; fi
    job_final_state "$1"
}

is_bad() {
    case "$1" in
        FAILED|NODE_FAIL|CANCELLED|TIMEOUT) return 0 ;;
        *) return 1 ;;
    esac
}

# ── checkpoint detection ─────────────────────────────────────────────────────
best_checkpoint() {
    # Prefer most-recent step checkpoint, fall back to epoch checkpoint
    if [[ -f "${OUTPUT_DIR}/last_step.pt" ]]; then
        echo "${OUTPUT_DIR}/last_step.pt"
    elif [[ -f "${OUTPUT_DIR}/last.pt" ]]; then
        echo "${OUTPUT_DIR}/last.pt"
    fi
}

epochs_done() {
    local mf="${OUTPUT_DIR}/metrics.json"
    [[ -f "$mf" ]] || { echo 0; return; }
    ${PYBIN} -c "import json; d=json.load(open('${mf}')); print(len(d))" 2>/dev/null || echo 0
}

# ── resubmission ─────────────────────────────────────────────────────────────
resubmit() {
    local ckpt config_to_use
    ckpt=$(best_checkpoint)

    if [[ -z "$ckpt" ]]; then
        log "ERROR: no checkpoint in ${OUTPUT_DIR} — cannot auto-recover"
        return 3
    fi

    log "Checkpoint: ${ckpt} (epochs_done=$(epochs_done))"

    # Patch the resume config to point at the actual checkpoint
    ${PYBIN} -c "
import yaml, pathlib
cfg = yaml.safe_load(pathlib.Path('${MASTERS}/configs/encoder/train/train_v2_15mm_k4_4gpu_resume.yaml').read_text())
cfg['training']['resume_from'] = '${ckpt}'
pathlib.Path('${MASTERS}/configs/encoder/train/train_v2_15mm_k4_4gpu_resume.yaml').write_text(yaml.dump(cfg))
"

    # Cancel any still-queued downstream job
    scancel "${J2}" 2>/dev/null || true

    # Resubmit the recovery continuation as new j1
    NEW_J1=$(sbatch --parsable \
        "${MASTERS}/ops/slurm/train_encoder_v2_15mm_k4_gh200_4gpu_j2.sbatch")
    NEW_J2=$(sbatch --parsable \
        --dependency=afterok:${NEW_J1} --kill-on-invalid-dep=yes \
        "${MASTERS}/ops/slurm/train_encoder_v2_15mm_k4_gh200_4gpu_j2.sbatch")

    log "Resubmitted → new_j1=${NEW_J1} new_j2=${NEW_J2} (retry ${RETRIES}/${MAX_RETRIES})"

    J1=${NEW_J1}
    J2=${NEW_J2}
    RETRIES=$((RETRIES + 1))
    PHASE="j1"
    DONE_PY=False
    write_state
    return 2
}

# ── main ─────────────────────────────────────────────────────────────────────
read_state
log "--- watcher tick | j1=${J1} j2=${J2} phase=${PHASE} retries=${RETRIES} done=${DONE} ---"

if [[ "${DONE}" == "True" ]]; then
    log "Training already marked complete — nothing to do."
    exit 0
fi

if [[ ${RETRIES} -ge ${MAX_RETRIES} ]]; then
    log "ALERT: max retries (${MAX_RETRIES}) reached — manual intervention needed."
    exit 3
fi

S1=$(job_state "${J1}")
S2=$(job_state "${J2}")
log "j1=${J1} state=${S1}  j2=${J2} state=${S2}"

EP=$(epochs_done)
log "Epochs completed so far: ${EP}/20"

DONE_PY=False

if is_bad "${S1}"; then
    log "ALERT: j1 (${J1}) is ${S1} — attempting auto-recovery"
    resubmit
    exit $?
fi

if [[ "${S1}" == "COMPLETED" ]]; then
    PHASE="j2"
    if is_bad "${S2}"; then
        log "ALERT: j2 (${J2}) is ${S2} — attempting auto-recovery"
        resubmit
        exit $?
    fi
    if [[ "${S2}" == "COMPLETED" ]]; then
        log "SUCCESS: both jobs completed. Epochs done: ${EP}/20"
        DONE_PY=True
        write_state
        exit 0
    fi
    log "j2 still ${S2} — waiting"
else
    log "j1 still ${S1} — waiting"
fi

# Write updated state (phase may have changed)
write_state
exit 0
