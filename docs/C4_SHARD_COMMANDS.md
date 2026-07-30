# C4 on this node — shard-by-shard commands

Companion to `RUNNING_ON_H100.md`. That doc assumes a node where you own every
GPU, so it says "run `scripts/run_c4.sh` and wait". This node is shared and
GPUs 4–7 are usually full, so the run is broken into **16 independent units you
launch by hand** whenever a GPU frees up.

Everything is already installed and downloaded — **skip Part A of
`RUNNING_ON_H100.md` entirely.** Jump straight to [Step 1](#step-1-launch-units).

---

## What is already set up

| Thing | Where | Note |
|---|---|---|
| PARA + EVA | `data/para`, `data/eva` | symlinks into `/shared/dataset` — no download needed |
| FLUX.1-Kontext-dev, Qwen2-VL-7B, CLIP ViT-L/14, DINOv2-base | `/shared/amin/hf_cache` | **not** the shared `/shared/huggingface` cache, which isn't writable by you (CLIP is owned by `bofeng`, the FLUX lock dir by `joe`) — using it fails with `PermissionError` |
| LAION aesthetic head | `~/.cache/syntheticaudience/aesthetic/` | |
| Python env | conda env `persona` | `scripts/c4_shard.sh` activates it for you |

**Do not rely on the inherited `HF_HOME`.** This account's profile exports
`HF_HOME=/shared/huggingface`, which is *not* writable by you — CLIP's cache
entry is owned by `bofeng` and the FLUX lock directory by `joe`, so loading a
model there dies with `PermissionError` on a lock file. `c4_shard.sh`
deliberately overrides it to `/shared/amin/hf_cache` (override with `C4_HF_HOME`
if you ever need to). If you invoke `script/c4_refine.py` **directly**, you must
export it yourself:

```bash
export HF_HOME=/shared/amin/hf_cache
```

---

## The aspect-ratio bug (fixed) — and why runs before it are worthless

`FluxKontextPipeline` takes its output canvas from the `width`/`height`
arguments, **not** from the reference image. `src/editor/flux_editor.py` never
passed them, so they defaulted to 1024x1024 and *every edit came back square*
while every source kept its native aspect. Both the aesthetic objective and the
DINOv2 drift metric then compared across that geometry change.

Measured cost of the distortion alone, before any content change:

| image | native | native score | as 1024x1024 | penalty |
|---|---|---|---|---|
| `eva__70724` | 630x470 | 5.045 | 4.579 | **-0.466** |
| `eva__103981` | 640x480 | 5.161 | 5.326 | +0.165 |
| `eva__602113` | 427x640 | 5.899 | 5.853 | -0.047 |

Arbitrary and image-dependent. On `eva__70724` it made improvement
arithmetically impossible — 10 steps, 30 candidates, best-so-far never moved off
5.045. It also *inflated apparent drift*: mean drift read 0.688, and "improving"
candidates were rejected at 0.36-0.63 as if they had destroyed the subject.

With `width`/`height` passed through (edits now come back e.g. 1184x880 for a
630x470 source), the same image gains **+0.376 with commits at steps 1, 2, 3**
and drift sits at 0.93-0.99 — the 0.78 cap barely binds.

**Consequence: any run produced before this fix should be discarded.** Its
trajectories are flat for reasons that have nothing to do with the critic being
tested.

### Edit boldness: keep the documented defaults

Once geometry is correct, boldness turns out not to matter much. An A/B on
identical images:

| config | gain on `eva__70724` | drift |
|---|---|---|
| `GUIDANCE=3.0` + emphasis (documented) | **+0.382** | 0.945-0.979 |
| `GUIDANCE=2.5`, `EMPHASIS=""` | +0.376 | 0.929-0.979 |

So `scripts/c4_shard.sh` keeps the pre-registered defaults and the bug fix is
the only deviation from `RUNNING_ON_H100.md`. The gentler variant is available
via `GUIDANCE=2.5 EMPHASIS="" scripts/c4_shard.sh ...` but tends to produce
near-invisible edits (drift 0.98-0.99) on some images.

Both settings are recorded in every run's `logs/c4_<condition>/*.json`
(`guidance_scale`, `edit_emphasis`), so a commit rate can always be traced back
to the parameters that produced it.

**Check drift early on any run.** After the first image or two:

```bash
python - <<'PY'
import json, glob, collections, os
for f in glob.glob('<OUTPUT_ROOT>/edits/*/_cache*.json'):
    d = json.load(open(f))
    dr = [v['drift'] for v in d.values()]
    if not dr: continue
    below = sum(1 for x in dr if x < 0.78)
    print(f"{f.split('/edits/')[1].split('/')[0]:12s} n={len(dr):4d} "
          f"drift mean={sum(dr)/len(dr):.3f} below-cap={below}/{len(dr)}")
PY
```

If most candidates sit below the cap, edits are too bold. If they all sit above
it but nothing commits, they are too timid. Either way, fix it before spending
the full ~16 h.

---

## Step 0 (optional): let the scheduler place everything

If you'd rather not babysit GPUs, this walks the whole unit list and starts each
one as soon as an allowed GPU has room:

```bash
cd /shared/amin/code/playground/SyntheticAudience
nohup env OUTPUT_ROOT=/shared/amin/c4_runs/c4_run1 GPUS=4,5,6,7 \
  scripts/c4_autorun.sh > /shared/amin/c4_runs/c4_run1/autorun.out 2>&1 &

tail -f /shared/amin/c4_runs/c4_run1/autorun.out
```

It prefers resident weights, only falling back to `CPU_OFFLOAD=1` after
`OFFLOAD_AFTER_SECS` (default 1800) of being unable to place anything; it scans
the whole queue rather than blocking on the head, so cheap `static` units still
get placed while an expensive `society` unit waits for a big GPU; and it
requeues any unit that OOMs or aborts. Placement chatter lands in
`$OUTPUT_ROOT/stdout/placement.log`.

Killing it is safe — every unit is `--resume`-based. **Run only one at a time**,
or two schedulers will place the same unit twice.

The rest of this document is the manual path, which does exactly the same thing
one command at a time.

---

## Step 1: launch units

One unit = one **condition** × one **quarter of the images**. Each is independent,
pinned to the GPU you name, and safe to re-run.

```bash
cd /shared/amin/code/playground/SyntheticAudience
export OUTPUT_ROOT=/shared/amin/c4_runs/c4_run1

scripts/c4_shard.sh <GPU> <CONDITION> <i/N>
```

The script refuses to start if the GPU lacks the VRAM for that unit, so you can
fire a command optimistically and it will just tell you to wait.

> **Give yourself more headroom than the minimum.** The other jobs on this node
> *grow* their allocation while running. Three separate launches during setup
> passed the free-memory check and then OOM'd 60–90 s later, mid-load, because a
> neighbour expanded (one went from 64 GB to 111 GB). `c4_shard.sh` samples free
> memory twice and takes the worse reading, which catches a GPU that is actively
> filling, but it cannot predict a neighbour that starts growing after you launch.
> In practice: prefer a GPU with **~80 GB free** for `society`/`blind` rather than
> the bare 62 GB minimum. An OOM costs you only the load time — the unit is
> resume-safe — but it costs you the GPU slot.

### The 16 units

Run them in any order, as many at a time as the GPUs allow. `society` and
`blind` are the expensive ones — start those first when you get a big GPU.

```bash
# society (needs ~62 GB free: FLUX + Qwen2-VL resident)
scripts/c4_shard.sh <GPU> society 0/4
scripts/c4_shard.sh <GPU> society 1/4
scripts/c4_shard.sh <GPU> society 2/4
scripts/c4_shard.sh <GPU> society 3/4

# blind (needs ~62 GB free)
scripts/c4_shard.sh <GPU> blind 0/4
scripts/c4_shard.sh <GPU> blind 1/4
scripts/c4_shard.sh <GPU> blind 2/4
scripts/c4_shard.sh <GPU> blind 3/4

# static (needs ~44 GB free — no VLM is loaded for this condition)
scripts/c4_shard.sh <GPU> static 0/4
scripts/c4_shard.sh <GPU> static 1/4
scripts/c4_shard.sh <GPU> static 2/4
scripts/c4_shard.sh <GPU> static 3/4

# reward_only (needs ~44 GB free — no VLM either)
scripts/c4_shard.sh <GPU> reward_only 0/4
scripts/c4_shard.sh <GPU> reward_only 1/4
scripts/c4_shard.sh <GPU> reward_only 2/4
scripts/c4_shard.sh <GPU> reward_only 3/4
```

To leave one running after you disconnect:

```bash
nohup scripts/c4_shard.sh 6 society 0/4 > /dev/null 2>&1 &
```

(its own log is written to `$OUTPUT_ROOT/stdout/` either way)

### If a GPU is close but not quite big enough

```bash
CPU_OFFLOAD=1 scripts/c4_shard.sh 6 society 0/4
```

This streams the FLUX weights instead of keeping them resident: **~25 GB less
VRAM, but a lot slower** — the transformer is copied host→device on every
`pipe()` call, i.e. three times per refinement step.

How much slower is not cleanly measured. The one timed run — the setup smoke
test — was `CPU_OFFLOAD=1` *and* sharing its GPU with a job at 100% utilisation,
and came out at **46 s/edit** (24 edits in 18m38s). Those two penalties are
confounded in that number. The ~7 s/edit figure for resident weights on a quiet
GPU is a *published-throughput estimate, not measured here* — no GPU on this
node was ever free enough to time it. Treat both ends of the range as soft until
you get a clean run.

Practical rule: offload is worth it for `static`/`reward_only` on a mid-size
GPU, and usually not worth it for `society`/`blind`. `FORCE=1` skips the VRAM
check altogether — expect an OOM if you're wrong.

---

## Step 2: check progress

```bash
scripts/c4_status.sh /shared/amin/c4_runs/c4_run1
```

Shows finished images per condition/shard, which workers are alive and on which
GPU, current free VRAM per GPU, and disk used. A condition is complete at
**100 images**.

Note: a unit's summary JSON is only flushed every 10 images, so a shard that
just started reads as `(not started)` until its first checkpoint — the "running
now" section is the live view.

Per-unit logs:

```bash
tail -f /shared/amin/c4_runs/c4_run1/stdout/c4_society_shard0of4_gpu6.log
```

---

## Step 3: resume anything that died

Re-run **the exact same command**. Finished images are skipped, and the
per-shard edit cache means already-generated FLUX candidates are not paid for
twice. There is no cleanup step and no risk of double-counting.

---

## Step 4: figures + table

Only after all 16 units report done:

```bash
cd scripts/analysis
python c4_trajectory.py  --output-root /shared/amin/c4_runs/c4_run1
python c4_qualitative.py --output-root /shared/amin/c4_runs/c4_run1
cd ../..
```

Produces `analysis/c4.json`, `analysis/c4_summary.md`, and six PNGs in
`analysis/figs/`: `c4_trajectory`, `c4_raw_objective`, `c4_headline`,
`c4_drift`, `c4_diversity`, `c4_qualitative`.

You can run these early on a partial set to sanity-check the plumbing — the
numbers will be meaningless but the figures will render. `c4_qualitative.py`
requires at least some `society` rows to exist.

---

## Step 5: zip and send

```bash
cd /shared/amin/c4_runs
zip -r c4_run1_full.zip c4_run1                      # everything (~25 GB)
zip -r c4_run1_logs.zip c4_run1/logs c4_run1/analysis # small, reproduces every number
```

---

## Budget

12,000 FLUX edits total (100 images × 4 conditions × 10 steps × 3 candidates).

| | per step (3 candidates) | source |
|---|---|---|
| `static`, resident weights, **quiet** GPU | **34 s** | measured, 5 consecutive steps (34/33/35/34 s) |
| `society`, resident weights, contended GPU | ~48 s | measured, 6 consecutive steps |
| `CPU_OFFLOAD=1` on a contended GPU | ~140 s | measured (24 edits in 18m38s) |

A FLUX edit costs **~11 s** either way — GPU contention barely moved it, so do
not expect a quiet GPU to be dramatically faster. (An earlier ~7 s figure in
this file was a published-throughput guess and was wrong.) On top of the three
edits, each step pays ~7 s for a `society` critique (10 personas in one batch),
~4 s for `blind`, ~2 s for the distiller, and nothing for
`static`/`reward_only`.

Per shard of 25 images x 10 steps: ~2.4 h for `static`/`reward_only`, ~2.8 h for
`blind`, ~3.0 h for `society` — **~42 GPU-hours for all 16 units**, so roughly
**11 h on four GPUs** or **14 h on three**. Add slack: GPUs hovering near the
memory threshold spend time rejecting units rather than running them. Output is
~25 GB of PNGs, going to `/shared` (4.3 TB free).

To re-measure at any point, take the delta in edited-PNG count over a few
minutes (`scripts/c4_status.sh` prints the count).

The single biggest lever is **not** parallelism — it is landing units on GPUs
that are quiet *and* have the memory to hold the weights resident. A unit
sharing a 100%-utilised GPU runs several times slower even when it fits.
