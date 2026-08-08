#!/bin/bash
# Poll until Slurm accepts submissions again (slurmdbd back up), then submit the
# two goal-reward SAC runs exactly once and exit. Safe to run in the background.
#
#   nohup bash V-Max/slurm/autosubmit_when_ready.sh >/dev/null 2>&1 &

cd /zfsauton2/home/yixiz/waymax_rs || exit 1
LOG=/zfsauton2/home/yixiz/waymax_rs/autosubmit_goal.log
echo "[$(date)] watcher started (polling every 60s, up to ~6h)" >> "$LOG"

for i in $(seq 1 360); do
  out1=$(env -u SLURM_JOB_ID -u SLURM_JOBID sbatch V-Max/slurm/train_sac_goal.sbatch reached_goal_once 2>&1)
  if [[ $? -eq 0 ]]; then
    echo "[$(date)] submitted reached_goal_once: $out1" >> "$LOG"
    out2=$(env -u SLURM_JOB_ID -u SLURM_JOBID sbatch V-Max/slurm/train_sac_goal.sbatch reached_goal 2>&1)
    echo "[$(date)] submitted reached_goal:      $out2" >> "$LOG"
    echo "[$(date)] DONE" >> "$LOG"
    exit 0
  fi
  echo "[$(date)] attempt $i not ready: $out1" >> "$LOG"
  sleep 60
done

echo "[$(date)] gave up after ~6h without a submission window" >> "$LOG"
exit 1
