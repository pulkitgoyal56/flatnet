---
title: Flatnet Models Summary
---

## Flatnet Comparison Table

|       Note        |                                        Serial | Type        | MNIST Size | Random Size | $\kappa$ | Epochs |
| :---------------: | --------------------------------------------: | ----------- | ---------- | ----------- | -------- | ------ |
|                   | [Flatnet 0 (lite)](./Flatnet%200%20(lite).md) | flatnetlite | 60000      | 0           | 0.0      | 8      |
|                   | [Flatnet 1 (lite)](./Flatnet%201%20(lite).md) | flatnetlite | 60000      | 0           | 0.2      | 8      |
|                   |                 [Flatnet 2](./Flatnet%202.md) | flatnet     | 60000      | 0           | 0.2      | 8      |
|                   |                 [Flatnet 3](./Flatnet%203.md) | flatnet     | 60000      | 0           | 0.4      | 8      |
|                   |                 [Flatnet 4](./Flatnet%204.md) | flatnet     | 60000      | 0           | 0.6      | 8      |
|                   |                 [Flatnet 5](./Flatnet%205.md) | flatnet     | 60000      | 0           | 0.8      | 8      |
|                   |                 [Flatnet 6](./Flatnet%206.md) | flatnet     | 60000      | 0           | 1.0      | 8      |
|                   |                 [Flatnet 7](./Flatnet%207.md) | flatnet     | 60000      | 10000       | 0.0      | 8      |
|                   |                 [Flatnet 8](./Flatnet%208.md) | flatnet     | 60000      | 10000       | 0.3      | 8      |
|                   |                 [Flatnet 9](./Flatnet%209.md) | flatnet     | 60000      | 10000       | 0.7      | 8      |
|                   |               [Flatnet 10](./Flatnet%2010.md) | flatnet     | 60000      | 10000       | 0.8      | 4      |
|                   |               [Flatnet 11](./Flatnet%2011.md) | flatnet     | 60000      | 10000       | 1.0      | 8      |
| Same as Flatnet 6 |               [Flatnet 12](./Flatnet%2012.md) | flatnet     | 60000      | 0           | 1.0      | 8      |
|         -         |               [Flatnet 13](./Flatnet%2013.md) | flatnet     | 60000      | 60000       | 0.0      | 8      |
|                   |               [Flatnet 15](./Flatnet%2015.md) | flatnet     | 60000      | 0           | 0.0      | 8      |

Random - Dataset of pixel-grid images with random pixel arrangements

## Complete Table

| -   | [CLIP_VIT_14](./CLIP_VIT_14.md) | CLIP  (ViT-L/14) | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < | < |
| --- |[CLIP_RN50x16]   | CLIP  (RN50x64) | <     | <          | <      | <           | <      | <           | <       | <            | <     | <          | <                     | <           | <      | <     | <     | <          | <        | <   | <   | <   | <               | <          | <           |

|       Note        |                                        Serial | Type        | MNIST Size | MNIST Test Size | iMNIST Size | iMNIST Test Size | Random Size | Random Test Size | iRandom | iRandom Test | Batch | Batch Test | $\kappa$ | Grid Images | Epochs | LR    | CPU   | Grid Cell Size | Grid Width | Grid Height | Show White Grid | Pixel Shape | Skeleton for MNIST Registration | Threshold Ratio for MNIST Registration | p_pixel(for Random generation) |
| :---------------: | --------------------------------------------: | ----------- | ---------- | --------------- | ----------- | ---------------- | ----------- | ---------------- | ------- | ------------ | ----- | ---------- | -------- | ----------- | ------ | ----- | ----- | -------------- | ---------- | ----------- | --------------- | ----------- | ------------------------------- | -------------------------------------- | ------------------------------ |
|                   | [Flatnet 0 (lite)](./Flatnet%200%20(lite).md) | flatnetlite | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | 256        | 0.0      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   | [Flatnet 1 (lite)](./Flatnet%201%20(lite).md) | flatnetlite | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | 256        | 0.2      | True        | 8      | 0.001 | True  | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |                 [Flatnet 2](./Flatnet%202.md) | flatnet     | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | 256        | 0.2      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |                 [Flatnet 3](./Flatnet%203.md) | flatnet     | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | 256        | 0.4      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |                 [Flatnet 4](./Flatnet%204.md) | flatnet     | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | 256        | 0.6      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |                 [Flatnet 5](./Flatnet%205.md) | flatnet     | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | 256        | 0.8      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |                 [Flatnet 6](./Flatnet%206.md) | flatnet     | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | 256        | 1.0      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |                 [Flatnet 7](./Flatnet%207.md) | flatnet     | 60000      | 10000           | 0           | 0                | 10000       | 10000            | 0       | 0            | 128   | 256        | 0.0      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |                 [Flatnet 8](./Flatnet%208.md) | flatnet     | 60000      | 10000           | 0           | 0                | 10000       | 10000            | 0       | 0            | 128   | 256        | 0.3      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |                 [Flatnet 9](./Flatnet%209.md) | flatnet     | 60000      | 10000           | 0           | 0                | 10000       | 10000            | 0       | 0            | 128   | 256        | 0.7      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |               [Flatnet 10](./Flatnet%2010.md) | flatnet     | 60000      | 10000           | 0           | 0                | 10000       | 10000            | 0       | 0            | 128   | 1          | 0.8      | True        | 4      | 0.001 | True  | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|                   |               [Flatnet 11](./Flatnet%2011.md) | flatnet     | 60000      | 10000           | 0           | 0                | 10000       | 10000            | 0       | 0            | 128   | 256        | 1.0      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
| Same as Flatnet 6 |               [Flatnet 12](./Flatnet%2012.md) | flatnet     | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | ~1         | 1.0      | True        | 8      | 0.001 | True  | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|         -         |               [Flatnet 13](./Flatnet%2013.md) | flatnet     | 60000      | 10000           | 0           | 0                | 60000       | 10000            | 0       | 0            | 128   | ~1         | 0.0      | True        | 8      | 0.001 | True  | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |
|    Odd one out    |                                    Flatnet 14 | flatnet     | 60000      | 10000           | 60000       | 10000            | 60000       | 10000            | 60000   | 10000        | 64    | ~1         | 0.0      | False       | 10     | 0.001 | False | 8              | 13         | 13          | False           | circles     | True                            | 0.0                                    | 0.030                          |
|                   |               [Flatnet 15](./Flatnet%2015.md) | flatnet     | 60000      | 10000           | 0           | 0                | 0           | 10000            | 0       | 0            | 128   | 256        | 0.0      | True        | 8      | 0.001 | False | 8              | 28         | 28          | False           | circles     | True                            | 0.0                                    | 0.045                          |

iMNIST - Inverted MNIST images dataset

## Model Params Comparison

- clip (ViT-L/14) - 427,616,513
- clip (RN50x16)  - 290,979,217
- flatnet         -  95,835,914
- flatnetlite     -  23,912,330

---
