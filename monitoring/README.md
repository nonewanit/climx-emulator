# Shared TensorBoard logs

Tracked copies of TensorBoard event files so teammates can view each other's
runs without any cloud service:

    git pull
    tensorboard --logdir monitoring/tensorboard --port 6006

Refresh after a run with:  make sync-tb && git commit -am "sync TB logs"

(Event files are a few MB per run — fine to track. The live per-machine logs
remain under the gitignored `results/runs/tensorboard/`.)

For cloud dashboards, use W&B instead (`logger.backend: wandb` in the
experiment config) from any network that can reach wandb.ai.
