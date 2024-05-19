|  M  |         Serial          | Name        | Random | $\lambda$ | Epochs |
|:---:|:-----------------------:| ----------- | ------ | --------- | ------ |
|  Y  | [[Flatnet 0 (lite)\|0]] | flatnetlite | 0      | 0.0       | 8      |
|  Y  | [[Flatnet 1 (lite)\|1]] | flatnetlite | 0      | 0.2       | 8      |
|  Y  |    [[Flatnet 2\|2]]     | flatnet     | 0      | 0.2       | 8      |
|  Y  |    [[Flatnet 3\|3]]     | flatnet     | 0      | 0.4       | 8      |
|  Y  |    [[Flatnet 4\|4]]     | flatnet     | 0      | 0.6       | 8      |
|  Y  |    [[Flatnet 5\|5]]     | flatnet     | 0      | 0.8       | 8      |
|  Y  |    [[Flatnet 6\|6]]     | flatnet     | 0      | 1.0       | 8      |
|  Y  |    [[Flatnet 7\|7]]     | flatnet     | 10000  | 0.0       | 8      |
|  Y  |    [[Flatnet 8\|8]]     | flatnet     | 10000  | 0.3       | 8      |
|  Y  |    [[Flatnet 9\|9]]     | flatnet     | 10000  | 0.7       | 8      |
|  Y  |   [[Flatnet 10\|10]]    | flatnet     | 10000  | 0.8       | 4      |
|  Y  |   [[Flatnet 11\|11]]    | flatnet     | 10000  | 1.0       | 8      |
|     |   [[Flatnet 12\|12]]    | flatnet     | 0      | 1.0       | 8      |
|     |   [[Flatnet 13\|13]]    | flatnet     | 60000  | 0.0       | 8      |
|  Y  |   [[Flatnet 15\|15]]    | flatnet     | 0      | 0.0       | 8      |

##### Table dump

```
|  -  | [[CLIP_VIT_14\|0]] | CLIP  (ViT-L/14) | <     | <          | <      | <           | <      | <           | <       | <            | <     | <          | <                     | <           | <      | <     | <          | <        | <   | <   | <   |
```

|  M  |         Serial          | Name        | MNIST | MNIST Test | iMNIST | iMNIST Test | Random | Random Test | iRandom | iRandom Test | Batch | Batch Test | Regularization Weight | Grid Images | Epochs | CPU   | p_pixel(1) | Skeleton | GCS | GW  | GH  |
|:---:|:-----------------------:| ----------- | ----- | ---------- | ------ | ----------- | ------ | ----------- | ------- | ------------ | ----- | ---------- | --------------------- | ----------- | ------ | ----- | ---------- | -------- | --- | --- | --- |
|  Y  | [[Flatnet 0 (lite)\|0]] | flatnetlite | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | 256        | 0.0                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  | [[Flatnet 1 (lite)\|1]] | flatnetlite | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | 256        | 0.2                   | True        | 8      | True  | 0.045      | True     | 8   | 28  | 28  |
|  Y  |    [[Flatnet 2\|2]]     | flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | 256        | 0.2                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  |    [[Flatnet 3\|3]]     | flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | 256        | 0.4                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  |    [[Flatnet 4\|4]]     | flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | 256        | 0.6                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  |    [[Flatnet 5\|5]]     | flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | 256        | 0.8                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  |    [[Flatnet 6\|6]]     | flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | 256        | 1.0                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  |    [[Flatnet 7\|7]]     | flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | 128   | 256        | 0.0                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  |    [[Flatnet 8\|8]]     | flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | 128   | 256        | 0.3                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  |    [[Flatnet 9\|9]]     | flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | 128   | 256        | 0.7                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|  Y  |   [[Flatnet 10\|10]]    | flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | 128   | 1          | 0.8                   | True        | 4      | True  | 0.045      | True     | 8   | 28  | 28  |
|  Y  |   [[Flatnet 11\|11]]    | flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | 128   | 256        | 1.0                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |
|     |   [[Flatnet 12\|12]]    | flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | ~1         | 1.0                   | True        | 8      | True  | 0.045      | True     | 8   | 28  | 28  |
|     |   [[Flatnet 13\|13]]    | flatnet     | 60000 | 10000      | 0      | 0           | 60000  | 10000       | 0       | 0            | 128   | ~1         | 0.0                   | True        | 8      | True  | 0.045      | True     | 8   | 28  | 28  |
|  X  |   [[Flatnet 14\|14]]    | flatnet     | 60000 | 10000      | 60000  | 10000       | 60000  | 10000       | 60000   | 10000        | 64    | ~1         | 0.0                   | False       | 10     | False | 0.030      | True     | 8   | 13  | 13  |
|  Y  |   [[Flatnet 15\|15]]    | flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | 128   | 256        | 0.0                   | True        | 8      | False | 0.045      | True     | 8   | 28  | 28  |

### Important Comparisons
#### Model Params

- clip (ViT-L/14) - 427,616,513
- clip (RN50x16)  - 290,979,217
- flatnet         -  95,835,914
- flatnetlite     -  23,912,330


---
## Complete Original Table (with inconsequential columns at the end removed)

| Name        | MNIST | MNIST Test | iMNIST | iMNIST Test | Random | Random Test | iRandom | iRandom Test | random | Batch | Batch Test | Epochs | LR    | Regularization Weight | CPU   | p_pixel(1) | Seed | Skeleton | Grid Images | GCS | GW  | GH  | Threshold Ratio | White Grid | Pixel Shape |
| ----------- | ----- | ---------- | ------ | ----------- | ------ | ----------- | ------- | ------------ | ------ | ----- | ---------- | ------ | ----- | --------------------- | ----- | ---------- | ---- | -------- | ----------- | --- | --- | --- | --------------- | ---------- | ----------- |
| flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 0.2                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 0.4                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 0.6                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 0.8                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 1.0                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | random | 128   | 1          | 4      | 0.001 | 0.8                   | True  | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 0.0                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 0.3                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 0.7                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 10000  | 10000       | 0       | 0            | random | 128   | 256        | 8      | 0.001 | 1.0                   | False | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            | random | 128   |            | 8      | 0.001 | 1.0                   | True  | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 0      | 0           | 60000  | 10000       | 0       | 0            | random | 128   |            | 8      | 0.001 | 0.0                   | True  | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
| flatnet     | 60000 | 10000      | 60000  | 10000       | 60000  | 10000       | 60000   | 10000        | random | 64    |            | 10     | 0.001 | 0.0                   | False | 0.030      | 5    | True     | False       | 8   | 13  | 13  | 0.0             | False      | circles     |
| flatnetlite | 60000 | 10000      | 0      | 0           | 0      | 10000       | 0       | 0            |        | 128   | 256        | 8      | 0.001 | 0.2                   | True  | 0.045      | 5    | True     | True        | 8   | 28  | 28  | 0.0             | False      | circles     |
