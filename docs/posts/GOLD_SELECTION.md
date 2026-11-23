# Choose what to label with embedding-based selection

## TL;DR

Use `GoldSelector` to choose samples that cover different embedding neighborhoods. Set a review budget in samples. Each sample counts once, even when it has several vectors.

## Table of contents

- [Pick search queries for annotation](#pick-search-queries-for-annotation)
- [Select robot trials by their 3D motion windows](#select-robot-trials-by-their-3d-motion-windows)
- [From vector picks to sample IDs](#from-vector-picks-to-sample-ids)
- [Set the sample budget](#set-the-sample-budget)
- [Choose the representation and the objective](#choose-the-representation-and-the-objective)

You can afford to label only a fraction of an unlabeled pool. When near-duplicates dominate, a random draw can spend much of the budget on examples that teach the model similar things. If you train a multilingual product search ranker, a queue dominated by common queries can hide regional product names and less common ways to describe the same item. Selecting for coverage can bring those queries to annotators when they form distinct neighborhoods.

Goldener compares vectors from a pretrained model and selects actual samples for review. The same workflow can build a monitoring queue for investigating unfamiliar inputs.

## Pick search queries for annotation

These six synthetic vectors stand in for unlabeled product-search queries in three neighborhoods. The IDs could be keys in your search logs. The selector writes a named assignment to a local Pixeltable table, and the query returns sample IDs that you can send to an annotation queue.

```python
from uuid import uuid4

import torch
from torch.utils.data import Subset

from goldener import GoldGreedyKCenterSelectionTool, GoldSelector

vectors = torch.tensor(
    [[0.0, 0.0], [0.1, 0.0], [4.0, 0.0], [4.1, 0.0], [0.0, 5.0], [0.0, 5.1]],
)
sample_ids = [101, 102, 201, 202, 301, 302]
rows = [
    {"idx": sample_id, "idx_vector": pos, "vectorized": vector}
    for pos, (sample_id, vector) in enumerate(zip(sample_ids, vectors))
]
pool = Subset(rows, range(len(rows)))

selector = GoldSelector(
    table_path=f"annotation_demo_{uuid4().hex}",
    selection_tool=GoldGreedyKCenterSelectionTool(device="cpu"),
)
table = selector.select_in_table(pool, select_size=3, value="annotate")
review_ids = selector.get_selection_indices(table, value="annotate", idx_key="idx")

assert {sample_id // 100 for sample_id in review_ids} == {1, 2, 3}
print(sorted(review_ids))
```

Greedy K-center picks one query from each of the three neighborhoods in this example. Once a neighborhood has a center, taking another near-duplicate usually contributes less coverage. An isolated corrupt input can also look far away, so inspect the selected examples before labeling.

## Select robot trials by their 3D motion windows

When reviewing a robot learning to place objects, a brief sideways correction can disappear in a trial-level average. A 3D motion tracker records the end effector's position over time. The synthetic vectors below represent two windows per trial, using lateral and vertical displacement as two example features. Each window has its own `idx_vector`, and both windows share the trial's `idx`. The budget is two trials from a new deployment, with an older trial kept outside the candidate pool.

```python
from uuid import uuid4

import torch
from torch.utils.data import Subset

from goldener import GoldGreedyKCenterSelectionTool, GoldSelector

trial_ids = [10, 10, 20, 20, 30, 30, 99, 99]
windows = torch.tensor(
    [
        [0.0, 0.0], [0.01, 0.0], [0.1, 0.0], [0.11, 0.0],
        [0.05, 0.0], [5.1, 0.0], [12.0, 0.0], [12.1, 0.0],
    ],
)
rows = [
    {"idx": trial_id, "idx_vector": pos, "vectorized": vector}
    for pos, (trial_id, vector) in enumerate(zip(trial_ids, windows))
]
trials = Subset(rows, range(len(rows)))
selector = GoldSelector(
    table_path=f"motion_review_{uuid4().hex}",
    selection_tool=GoldGreedyKCenterSelectionTool(device="cpu"),
)
table = selector.select_in_table(
    trials,
    select_size=2,
    value="review_motion",
    restrict_to={10, 20, 30},
    restriction_idx_key="idx",
)

review_ids = selector.get_selection_indices(table, value="review_motion")
window_ids = selector.get_selection_indices(table, value="review_motion", idx_key="idx_vector")
assert review_ids == {10, 30}
assert window_ids == {0, 1, 4, 5}
print(sorted(review_ids))
```

Trial 30 stands out because its second window contains a large lateral correction. Once selected, both windows are marked. Use the trial IDs to queue the original 3D trajectories for review and the window IDs to look up their stored vectors.

## From vector picks to sample IDs

Greedy K-center picks centers from the candidate vectors to cover the embedding space. Once it has a center, each new pick maximizes the distance to the nearest center. Later rounds can use vectors from already-selected samples as anchors. When candidate vectors are identical, the tool can return the same position more than once.

`GoldSelector` maps those tensor positions to distinct sample IDs (`idx`) and marks every stored vector row for each selected sample. It can make another pass when several picks resolve to the same sample. Use `get_selection_indices` for the sample IDs, or pass `idx_key="idx_vector"` to retrieve the vector IDs.

## Set the sample budget

Pass an integer `select_size` for a sample count or a float for a fraction of the distinct samples in scope, rounded up. Use `restrict_to` to define that scope. Its IDs refer to vectors by default. Pass `restriction_idx_key="idx"` to restrict by sample ID, as in the motion example.

With `label_key`, the selector tries to allocate at least one sample to every label on the first pass and raises a `ValueError` if it cannot. If samples have overlapping labels, the selection can exceed the requested count. Check the number of returned sample IDs when the budget is strict.

Selections already saved under the same `value` count toward a repeated request. Samples assigned to a different value are unavailable. Use a fresh result table when the representation or selection recipe changes.

## Choose the representation and the objective

Use a pretrained model through `GoldTorchEmbeddingTool` and `GoldDescriptor` to extract embeddings. Use `GoldTensorVectorizationTool` in the descriptor, or `GoldVectorizer` on stored embeddings, to turn model features into vectors. Point the selector's `vectorized_key` at the resulting column. The embedding layer and distance metric determine what looks redundant to the selector.

Greedy K-center emphasizes coverage. `GoldGreedyKernelPointsSelectionTool` aims to approximate a distribution in its chosen kernel space. For a large pool, chunking limits the per-chunk computation and makes coverage decisions within smaller pools. Reducing vector width also saves work, with the risk of erasing distinctions you need. [Coreset selection survey](https://arxiv.org/abs/2505.17799) · [Reducing the dimensionality of embeddings](GOLD_REDUCTION.md)

For monitoring, review the selected IDs to discover failure modes and collect examples for follow-up. Coverage selection changes the mixture you inspect, so draw a separate random audit sample from the deployment population when estimating deployment error rates. If the samples are already labeled and you need disjoint training and evaluation sets, use the same selection stage inside `GoldSplitter`. [Splitting labeled data with Goldener](GOLD_SPLIT.md)

Sponsored by [Pixeltable](https://www.pixeltable.com/).
