# DEPRECATED — Minerva uses LSF (bsub), not SLURM (sbatch).
#
# This SLURM template does not apply to the actual compute target. The GPU nodes
# are NVIDIA B200 (Blackwell sm_100) scheduled by IBM LSF.
#
#   Use:  jobs/finetune.lsf   (bsub batch script)
#   See:  SETUP.md  section B (Minerva — B200 / x86_64 / LSF)
#
# Kept only as a historical reference. Do not submit with sbatch.
