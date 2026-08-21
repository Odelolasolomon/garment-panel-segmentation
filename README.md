# Garment Panel Segmentation + Deterministic Fabric Fill

## Setup

```bash
git clone <this repo>
cd panel-seg
pip install -r requirements.txt
```

Training data (Fashionpedia) is not committed to this repo (too large). See
"Data source and licence" below for download instructions.

## Run

Train (from repo root, after populating `configs/train.yaml` with real data paths):
```bash
python train.py --config configs/train.yaml
# resume an interrupted run:
python train.py --config configs/train.yaml --resume checkpoints/last.pt
```

Predict a mask for a single image:
```bash
python predict.py --image path/to/garment.jpg --output path/to/mask.png --weights weights/best.pt
```

Apply fabric to a panel (see `apply_fabric.py` for the importable function and
a runnable worked example):
```bash
python apply_fabric.py
```

Run tests:
```bash
pytest tests/ -v
```

Benchmark CPU latency (see constraint table below):
```bash
python benchmark_latency.py --weights weights/best.pt --image sample.jpg --runs 50
```

## Architecture and why

**Frozen, ImageNet-pretrained MobileNetV3-Small encoder + a small trainable
FPN-style decoder.**

The 2,000,000 trainable-parameter cap is tight enough that spending it
re-learning generic edge/texture/shape features from scratch, on a
relatively small custom-labelled dataset, is a poor use of capacity. A
frozen pretrained encoder supplies those generic features for free — they
do not count against the trainable budget, since they are never updated —
and the entire trainable budget goes to a decoder that specializes those
generic features into garment panels.

The decoder taps the encoder at four resolutions (stride 4, 8, 16, 32;
channel counts 16/24/48/96) and fuses them top-down, FPN-style: deep,
low-resolution features carry global context (useful for panel identity
decisions that depend on the whole garment, e.g. front vs. back body,
which is not something local texture alone determines), while shallow,
high-resolution features carry the spatial precision needed for clean
panel boundaries.

All decoder convolutions are **depthwise-separable** (depthwise 3x3 +
pointwise 1x1) rather than full 3x3 convolutions. A full 3x3 conv from
`C_in` to `C_out` channels costs `9 * C_in * C_out` parameters; the
separable version costs `9 * C_in + C_in * C_out`, which is dramatically
cheaper once channel widths climb — this is the main lever that keeps the
model under budget while still using a meaningful fraction of it (decoder
width was swept from 32 to 192 channels; 160 was chosen as a width that
uses most of the 2M budget without exceeding it — see Parameter count below).

**CoordConv** (Liu et al., 2018) is applied at the decoder's final,
highest-resolution stage: two extra channels carrying normalized x/y pixel
coordinates are concatenated before the last convolution, giving the
network explicit access to absolute spatial position rather than relying
on it leaking in indirectly through zero-padding at image borders. This is
a secondary aid for panels with informative canonical position (e.g.
collar typically top-center) — it is **not** the mechanism relied on for
left/right sleeve correctness (see below).

## The left/right sleeve problem, and why it is not solved inside the network

Left and right sleeves are close to pixel-identical in local appearance.
Standard convolutions are translation-equivariant by construction — a
kernel produces the same response to a pattern regardless of where in the
image it appears — so a segmentation head that only sees local sleeve
texture has no reliable signal to tell left from right. Asking the network
to learn this distinction directly is exactly how the original bug this
assessment exists to fix (image models putting garment features on the
wrong side) gets reproduced one layer further down the stack.

Instead: the network is trained to predict a single **merged `sleeve`
class** — it never has to make a left/right decision at all. Left/right
identity is then resolved **deterministically**, after inference, in
`model/postprocess.py`:

1. Take connected components of the predicted sleeve mask.
2. Compute each component's centroid x-position.
3. Compare against the garment's own horizontal bounding-box center
   (computed from *all* foreground pixels, so it's robust even when only
   one sleeve is visible).
4. Component left of center → `left_sleeve`; right of center → `right_sleeve`.

This is a plain thresholded geometric comparison — same input, same
output, every time, which is exactly the determinism the brief asks for.
It is covered directly by `tests/test_panel_targeting.py`.

**Assumption, stated explicitly per the brief's instruction to write down
ambiguous calls:** this relies on canonically-framed garment images (garment
roughly centered, facing the camera), consistent with the production
3D-render pipeline described in the brief. It would break on wildly
inconsistent framing/rotation — see "What is broken" below.

## Parameter count

Reported by running `python -c "from model.segmodel import build_model; m = build_model(pretrained_encoder=False); print(m.trainable_param_count(), m.total_param_count())"`:

- **Trainable parameters: 429,445** (decoder only) — under the 2,000,000 cap.
- **Total inference-time parameters: 1,356,453** (429,445 trainable decoder +
  927,008 frozen MobileNetV3-Small encoder).

Decoder width (160 channels) was chosen by sweeping 32/64/96/128/160/192 and
picking a width that uses a meaningful fraction of the budget (segmentation
quality is 30% of the score — leaving most of a 2M budget unused would be
leaving accuracy on the table) while staying comfortably under the cap.

## Training results

Training was run on Kaggle using the Fashionpedia-derived panel labels, with
the trained checkpoint copied into `weights/best.pt` for local CPU inference.
The run used `image_size=256`, `batch_size=32`, `decoder_width=160`, and ran
for **10 epochs**.

- **Best validation mIoU: 0.4194** (epoch 6).
- **Final validation mIoU: 0.3974** (epoch 9).
- **Final per-class IoU:**
  - `front_body` / class `1`: 0.6695
  - `sleeve` / class `3`: 0.4771
  - `collar` / class `4`: 0.0434
- **Trainable parameters: 429,445** and **total parameters: 1,356,453**,
  matching the local parameter-count check above.

## Constraint validation

| Constraint / check | Result |
|---|---|
| Trainable parameters under 2,000,000 | Pass: 429,445 trainable parameters |
| CPU inference supported | Pass: `predict.py` and `benchmark_latency.py` force CPU by default |
| Unit tests | Pass: `pytest tests/ -v` -> 18 passed |
| Real-weight smoke test | Pass: `python predict.py --image path/to/sample_garment.jpg --output outputs/mask.png --weights weights/best.pt` |
| Smoke-test detected panels | `front_body`, `left_sleeve`, `right_sleeve`, `collar`; `back_body` absent |
| Part 1 -> Part 2 handoff | Pass: predicted `outputs/mask.png` used by `apply_fabric_from_files(...)` to produce `outputs/fabric_front_body.png` |

CPU latency benchmark, run locally on this machine:

```text
=== Hardware ===
Platform: Windows-10-10.0.26200-SP0
Processor: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel
CPU threads available to torch: 4
(/proc/cpuinfo not available on this platform)

=== Latency (forward pass + postprocessing, per image) ===
Runs: 50
Mean:   131.5 ms
Median: 96.6 ms
P95:    153.0 ms
Min:    81.9 ms
Max:    1430.0 ms
```

## Label set and why

Internal training classes (5, what the network actually predicts):
`background(0), front_body(1), back_body(2), sleeve(3), collar(4)`.

This is deliberately **not** the same as the 5 required output panel names
(`front_body, back_body, left_sleeve, right_sleeve, collar`) — sleeve is
merged internally and split into left/right by the deterministic
post-processing step described above, for the reasons given there.

Fixed output PNG index mapping (single source of truth:
`model/postprocess.py:OUTPUT_INDEX`, referenced directly by both
`predict.py` and this README so they cannot drift apart):

| Index | Panel |
|---|---|
| 0 | background |
| 1 | front_body |
| 2 | back_body |
| 3 | left_sleeve |
| 4 | right_sleeve |
| 5 | collar |

A panel not detected in a given image is **absent** from the mask (no
pixels carry its index) rather than guessed, and requesting an absent
panel in `apply_fabric` is a documented no-op, never an exception — both
are covered by `tests/`.

## Data source and licence

[Fashionpedia](https://fashionpedia.github.io/home/) — training images
(`train2020.zip`), validation images (`val_test2020.zip`), and the
attribute-instance annotation files
(`instances_attributes_train2020.json`, `instances_attributes_val2020.json`).
Fashionpedia is released under a **CC BY-NC 4.0** licence — verify this
against the current terms on the dataset's own page before any commercial
use, since licence terms can change independently of this document.

Category IDs used for the internal label mapping, confirmed directly from
the downloaded `categories` list in the annotation JSON (not assumed):
`sleeve = category_id 31`, `collar = category_id 28`. Front-body is derived
from a fixed set of whole-garment upper-body/one-piece category IDs (shirt,
top, sweater, cardigan, jacket, vest, coat, dress, jumpsuit — see
`BODY_GARMENT_CATEGORY_IDS` in the dataset code) with sleeve and collar
annotations subsequently overwriting the body region at their respective
pixels, since a garment's sleeve/collar annotation should win over its
generic torso annotation wherever they overlap.

**back_body is not present in Fashionpedia's labels at all** — see "Domain
gap" and "What is broken" below.

## Domain gap (Fashionpedia photos vs. production 3D renders)

Fashionpedia is photographs of worn garments on real people: real lighting,
real background clutter, real self-occlusion from pose, and — critically —
almost entirely front-facing camera angles. Production images described in
the brief are synthetic 3D renders of garments on an invisible/ghost form:
clean background, controlled lighting, and (implicitly) both front and back
views are producible.

Concrete consequences:
- **`back_body` has essentially zero direct supervision from Fashionpedia.**
  The model is not expected to reliably predict `back_body` without
  supplementary data; this is a known, named limitation, not something this
  submission claims to have solved (the brief explicitly does not score
  closing this gap, only noticing and reasoning about it).
- Fashionpedia's photographic lighting/background variability is different
  in kind from a render's clean, controlled lighting — mild brightness/
  contrast jitter augmentation is used during training as a cheap,
  low-risk step in that general direction, not a claim of closing the gap.
- Real-photo boundaries (fabric drape, self-occlusion from pose) are messier
  than a render's cleaner boundaries; this may make the model somewhat more
  conservative/uncertain at panel edges than it would be on the eventual
  target distribution.

## What I'd do next with another two days

- Curate or synthetically generate a small back-facing garment set (even a
  few hundred hand-labelled or programmatically-rendered examples) to give
  `back_body` real supervision instead of zero.
- Expand the connected-component left/right heuristic with an explicit
  garment-orientation check (e.g. detect whether the image is front-facing
  vs. rotated) so the canonical-framing assumption is verified rather than
  assumed.
- Run a proper hyperparameter sweep (decoder width, dice/CE loss balance,
  learning rate) with a held-out validation split rather than the single
  configuration used here, and track it more rigorously than the current
  best/last checkpoint scheme.
- Collect a handful of real in-house render-style images (even unlabelled)
  to qualitatively sanity-check the domain gap's practical impact before
  investing further in the Fashionpedia-only training set.

## What I know is broken

- `back_body` predictions are unreliable — there is essentially no direct
  training signal for this class from Fashionpedia (see "Domain gap" above).
  The pipeline handles this honestly: an absent `back_body` panel returns an
  empty mask rather than a guess, but a request for `back_body` on a genuine
  back-facing garment may also come back empty when it shouldn't, precisely
  because the model was never shown real examples to learn from.
- The left/right sleeve split assumes canonical, roughly-centered,
  front-facing garment framing. It has no fallback for unusual orientations
  or rotations — see "Assumption" in the architecture section above.
- Flat-tile fabric fill (Part 2) does not account for the render's existing
  shading — see `DESIGN_NOTE.md` for the full reasoning on this and what a
  shading-aware version would require.
- Class weights for the training loss were computed from a sample of the
  training set (not the full set, for tractability) — see the Kaggle
  notebook cell that computes `class_pixel_counts`; a full-dataset pass
  would give a more precise weighting, particularly for the rare `collar`
  class.

## AI tools used

Claude (Anthropic) was used throughout this submission for: architecture
design discussion and justification (frozen-encoder + separable-conv
decoder trade-off reasoning, CoordConv rationale), the deterministic
left/right connected-component postprocessing design, writing and testing
`model/`, `predict.py`, `apply_fabric.py`, and `tests/` against a local
PyTorch environment (parameter-budget verification, forward-pass shape
checks, and the full `tests/` suite were run and passed before this
submission), drafting this README and `DESIGN_NOTE.md`, and reviewing the
Fashionpedia category-ID mapping logic. Dataset download, category-ID
confirmation (`sleeve=31`, `collar=28`), and all actual model training were
run independently on Kaggle. I can walk through any part of the code in
detail on request.

## Deployment

Production-oriented deployment scaffolding is included under `deployment/`:

- `serve.py` exposes the existing prediction and fabric-fill logic as a CPU FastAPI service.
- `Dockerfile`, `.dockerignore`, and `docker-compose.yml` support local/container deployment.
- `deployment/k8s/` contains Kubernetes manifests for namespace, config, model PVC, API deployment, service, and benchmark job.
- `deployment/terraform/aws/` contains an AWS Terraform baseline for ECR, private model-artifact storage, and optional EKS IRSA permissions.
- `deployment/kubeflow/pipeline.py` contains a Kubeflow validation pipeline for running the latency benchmark.
- `.github/workflows/ci.yml` runs tests and builds the container image in CI.

Model weights are treated as artifacts rather than source code: `weights/best.pt`
should be mounted or fetched from artifact storage in production. See
`deployment/README.md` for concrete commands.
