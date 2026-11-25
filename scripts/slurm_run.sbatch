#!/bin/bash
# Run the full JGI integration workflow (including the explorer bundle) as a slurm job.
#
#   sbatch scripts/slurm_run.sbatch /path/to/input_data /path/to/output_data
#
# The container image is the same one used for JupyterLab. Pick the runtime your HPC
# provides (set RUNTIME): apptainer/singularity (most centers), podman-hpc or shifter (NERSC).
# Run on a compute node, not a login node: hierarchical clustering and correlation steps
# are memory-heavy (a 25k-feature correlation-distance clustering needs roughly 5-10 GB).
#
#SBATCH --job-name=jgi-integration
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=jgi-integration-%j.log

set -euo pipefail
INPUT=${1:?usage: sbatch slurm_run.sbatch <input_data dir> <output_data dir>}
OUTPUT=${2:?usage: sbatch slurm_run.sbatch <input_data dir> <output_data dir>}
IMAGE=${IMAGE:-docker://ghcr.io/bkieft-usa/jgi-integration:latest}
RUNTIME=${RUNTIME:-apptainer}
EXTRA_ARGS=${EXTRA_ARGS:-}          # e.g. "--stop-after group_features" or "--start-at enrich"

CMD="python -m tools.cli run --config-dir /home/jovyan/work/input_data/config \
     --log-file /home/jovyan/work/output_data/run_${SLURM_JOB_ID:-local}.log ${EXTRA_ARGS}"

case "$RUNTIME" in
  apptainer|singularity)
    $RUNTIME exec --cleanenv \
      --bind "$INPUT":/home/jovyan/work/input_data --bind "$OUTPUT":/home/jovyan/work/output_data \
      --pwd /home/jovyan/work "$IMAGE" bash -lc "$CMD" ;;
  podman-hpc)
    podman-hpc run --rm -v "$INPUT":/home/jovyan/work/input_data -v "$OUTPUT":/home/jovyan/work/output_data \
      -w /home/jovyan/work "${IMAGE#docker://}" bash -lc "$CMD" ;;
  shifter)
    shifter --image="${IMAGE#docker://}" \
      --volume="$INPUT":/home/jovyan/work/input_data --volume="$OUTPUT":/home/jovyan/work/output_data \
      bash -lc "cd /home/jovyan/work && $CMD" ;;
  *) echo "Unknown RUNTIME=$RUNTIME" >&2; exit 2 ;;
esac

echo "Done. To explore: copy <output>/<project>/Dataset_Processing--*/Analysis--*/explorer/ to your laptop and run"
echo "  python -m tools.cli explore --bundle <that folder>   (or use an SSH tunnel: ssh -L 8050:localhost:8050 <host>)"
