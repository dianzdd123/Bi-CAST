# Bi-CAST · Bimanual magnetic micromanipulation

How much control should each hand delegate as a magnetic millirobot approaches a vessel wall or turns through a branch? Bi-CAST estimates a separate autonomy weight for each manipulator from microscope images, grip-force history and spatial risk. The navigation planner supplies a reference trajectory. The learned model determines how strongly the autonomous command contributes to each arm’s motion.

**Paper.** Yongchen Wang and Kangyi Lu (equal contribution), Lan Wei and Dandan Zhang. *Context-Aware Adaptive Shared Control for Magnetically-Driven Bimanual Dexterous Micromanipulation* (2026). [Manuscript](https://arxiv.org/abs/2603.14388v1) · [Full text with figures and tables](https://arxiv.org/html/2603.14388v1)

## From the paper to the source

| Paper component | Implementation | What to look for |
|---|---|---|
| Physical system, Fig. 2 and §III-B | [CAD files](cad/) | Native SOLIDWORKS robot part and maze/robot assembly for the millirobot and its experimental workspace. |
| Operator intent, Eqs. 4–5 | [Force preprocessing](tools/data/import_wall_distance_fsr.py) | Scale each hand’s force using its session-specific resting and override grips, then smooth over three samples. |
| Shared control, Eqs. 6–8 and Fig. 3 | [Model](src/iros/models/bicast.py) · [Inference example](examples/bicast_inference.py) | Four frames feed a Transformer. Two heads predict five future authority values per arm. Overlapping forecasts are combined causally before blending operator and planner commands. |
| Reference navigation | [Planner](src/iros/navigation/path_planner.py) · [Trajectory generator](src/iros/navigation/trajectory_generator.py) | Plan a route through registered free space, then compute robot-centre and endpoint waypoints. Arm-role switching in the planner and learned authority weighting are separate operations. |
| Supervised learning and evaluation, §IV | [Training](training/core/train_model.py) · [Evaluation](training/core/evaluate_models.py) | Train on Tasks 1 and 3, then use separate temporal portions of Task 2 for validation and testing. Table II compares network architectures. |
| Design ablations, Table III | [Ablation runner](training/scripts/run_table_iv_ablations.py) | Vary backbone, input streams, smoothness penalty or prediction horizon. The script and its JSON output retain the original `table_iv` filename. |

## CAD and experimental geometry

| File | Purpose and use |
|---|---|
| [IROS-robot.SLDPRT](cad/IROS-robot.SLDPRT) | Individual robot part. Open it directly in SOLIDWORKS to inspect its features and measure its geometry. Section III-B describes the physical robot as 4 mm long, with two 1 mm diameter × 1 mm magnets embedded at opposite ends in antiparallel orientation. These are paper specifications. |
| [maze_robot.SLDASM](cad/maze_robot.SLDASM) | Assembly supplied for the robot and maze. Keep both native files together and open the assembly to inspect their spatial relationship. Use **File → Find References** in SOLIDWORKS to check component resolution. The archive contains no separate maze part or neutral CAD export. A fully resolved rebuild has not been verified. |

The planner uses camera imagery and registered maze coordinates. The [maze image](assets/maze.png) and [robot template](assets/robot_template.jpg) are image-processing assets. The CAD assembly is for inspecting mechanical geometry. Converting a CAD distance to pixels requires the experiment’s image registration.

## What you can run

With the training extras installed, run the inference example from the repository root:

```bash
python examples/bicast_inference.py
```

The example uses random inputs and an untrained model. It prints left and right autonomy weights and blended command norms over eight steps. A weight of zero passes through the operator’s command. At the model’s upper bound of 0.9, the operator still contributes 10%.

For a recorded experiment, start with the [training and data guide](training/README.md). Navigation export requires a tracking CSV and maze registration. Model training additionally requires images, calibrated force signals and bilateral authority labels. The release includes code, configuration, image assets and CAD. Raw study recordings, trained weights, the Unity project and the physical haptic-control runtime are outside this archive. The eight-participant study in §IV-E reports results obtained on the physical apparatus.

## Project layout

```text
Bi-CAST/
├── cad/                      # Native SOLIDWORKS part and assembly
├── src/iros/                 # Installable Python package
│   ├── navigation/           # Planning, trajectories, and visualization
│   ├── navigation_experiments/
│   ├── maze/                 # Maze feature extraction
│   └── models/               # Bi-CAST authority-negotiation network
├── configs/navigation/       # Navigation JSON configurations
├── examples/                 # Small runnable examples
├── tools/                    # Annotation and data tools
├── training/                 # Training code and orchestration scripts
├── docs/                     # Focused technical documentation
├── data/                     # Local data (ignored by Git)
└── outputs/                  # Generated results (ignored by Git)
```

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional dependencies only for the features you need:

```bash
pip install -e '.[vision]'    # OpenCV and maze feature extraction
pip install -e '.[training]'  # PyTorch model training
pip install -e '.[ui]'        # Gradio annotation tools
pip install -e '.[dev]'       # Tests and linting
```

## Data and output paths

Navigation datasets are read from `data/navigation/<dataset-name>` by default.
Each dataset is expected to contain:

```text
<dataset-name>/
├── tracking_results.csv
└── registration_output/
    ├── maze_params.json
    └── maze_features.pkl
```

Keep large data outside the repository by setting `IROS_DATA_ROOT` to the
directory that contains the navigation datasets. Set `IROS_OUTPUT_ROOT` to
redirect generated files. For example:

```bash
export IROS_DATA_ROOT=/path/to/navigation-datasets
export IROS_OUTPUT_ROOT=/path/to/results
```

### Paper-aligned data preparation

Tracking videos are subsampled every other frame by default (`--interval 2`),
matching Sec. IV-B.  Authority FSR preprocessing uses the per-session natural
grip / deliberate-override calibration from Eq. (4), followed by the `w=3`
trailing average from Eq. (5); the importer also assembles
`fsr_L/fsr_R/iou/d_wall` into `alpha_labels.pkl`.  See
[training/README.md](training/README.md) for the exact commands.

## Common commands

```bash
# With a prepared navigation dataset, export a planned trajectory
iros-export-trajectory \
  --dataset YOUR_DATASET \
  --target A \
  --config configs/navigation/default.json

# Visualize a trajectory without opening a window
iros-visualize \
  --dataset YOUR_DATASET \
  --target A \
  --config configs/navigation/default.json \
  --no-show

# Run the lightweight test suite
pytest
```

See [navigation.md](docs/navigation.md),
[trajectory_export.md](docs/trajectory_export.md),
[bicast_model.md](docs/bicast_model.md), and
[training_parameters.md](docs/training_parameters.md) for subsystem details.


