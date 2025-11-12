import wandb
import pandas as pd
import numpy as np

ENTITY_PROJECT = "hyang20-univeristy-of-alberta/ddqn.action-encoding"
RUN_NAME = "PongNoFrameskip-v4__dqn__0__251111-173728"  # use the run name
METRIC_KEY = "train/train/time_elapsed_stepwise"
STEP_KEY = "global_step"

api = wandb.Api()
runs = api.runs(ENTITY_PROJECT)

# Find the run by name
run = next((r for r in runs if r.name == RUN_NAME), None)
if run is None:
    raise SystemExit(f"Run with name '{RUN_NAME}' not found in {ENTITY_PROJECT}")

rows = [row for row in run.scan_history(page_size=1000)]
df = pd.DataFrame(rows)

if METRIC_KEY not in df.columns:
    raise SystemExit(f"Metric '{METRIC_KEY}' not found. Available columns:\n{df.columns.tolist()}")
if STEP_KEY not in df.columns:
    df[STEP_KEY] = df.index

df = df[[STEP_KEY, METRIC_KEY]].dropna()
steps = df[STEP_KEY].to_numpy(dtype=np.float64)
times = df[METRIC_KEY].to_numpy(dtype=np.float64)

print(f"=== All training time data for run {RUN_NAME} ===")
print(f"Total records: {len(steps)}\n")

for i, (s, t) in enumerate(zip(steps, times)):
    print(f"[{i:03d}] step={s:.0f}, train_time={t/3600:.3f}h ({t:.1f}s)")
