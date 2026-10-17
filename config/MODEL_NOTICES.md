# Public pretrained models

The setup downloads only the three public checkpoints in `pretrained_models.json`. That manifest pins their source URLs, byte counts and SHA-256 checksums. Cached files stay under ignored `artifacts/weights/`; no project images or GT are uploaded. No external training dataset is downloaded by setup.

| Component | Selected artifact | Upstream notice |
|---|---|---|
| DISK | Depth-supervised `depth-save.pth` | [Author repository](https://github.com/cvlab-epfl/disk); [Apache-2.0 notice](licenses/DISK-LICENSE.txt) |
| LightGlue | DISK-specific `v0.1_arxiv` checkpoint | [Author release](https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv); [Apache-2.0 notice](licenses/LightGlue-LICENSE.txt) |
| EigenPlaces | ResNet18, 512-dimensional `v1.0` checkpoint | [Author repository and SF-XL training description](https://github.com/gmberton/EigenPlaces); [MIT source notice](licenses/EigenPlaces-LICENSE.txt) |
| hloc | Known-pose mapping code at the revision in `pyproject.toml` | [Author repository](https://github.com/cvg/Hierarchical-Localization); [Apache-2.0 notice](licenses/hloc-LICENSE.txt) |

LightGlue and hloc code are installed from immutable source revisions in the dependency lock. DISK is supplied through pinned Kornia/LightGlue. EigenPlaces architecture integration is still assigned to the vision developer; its checkpoint is cached and decoded by the smoke check, not yet used for place retrieval.

The eventual submission should declare these public pretrained models and distinguish their upstream training from fitting the team's landmark map on the allowed training interval. Preserve these notices when vendoring author code. This file does not change organizer data rights or the upstream model/dataset terms.
