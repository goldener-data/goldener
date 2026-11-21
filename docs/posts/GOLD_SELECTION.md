# Choose what to label with embedding-based selection

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

## Select recordings by their audio windows

A short sound can disappear in a recording-level average. The synthetic audio vectors below describe two windows per recording. Each window has an `idx_vector`, and both windows share the recording's `idx`. The review budget is two recordings from a new deployment cohort. The older recording is available in the dataset but outside this request.

```python
from uuid import uuid4

import torch
from torch.utils.data import Subset

from goldener import GoldGreedyKCenterSelectionTool, GoldSelector

recording_ids = [10, 10, 20, 20, 30, 30, 99, 99]
windows = torch.tensor(
    [
        [0.0, 0.0], [0.01, 0.0], [0.1, 0.0], [0.11, 0.0],
        [0.05, 0.0], [5.1, 0.0], [12.0, 0.0], [12.1, 0.0],
    ],
)
rows = [
    {"idx": recording_id, "idx_vector": pos, "vectorized": vector}
    for pos, (recording_id, vector) in enumerate(zip(recording_ids, windows))
]
recordings = Subset(rows, range(len(rows)))
selector = GoldSelector(
    table_path=f"audio_review_{uuid4().hex}",
    selection_tool=GoldGreedyKCenterSelectionTool(device="cpu"),
)
table = selector.select_in_table(
    recordings,
    select_size=2,
    value="listen",
    restrict_to={10, 20, 30},
    restriction_idx_key="idx",
)

review_ids = selector.get_selection_indices(table, value="listen")
window_ids = selector.get_selection_indices(table, value="listen", idx_key="idx_vector")
assert review_ids == {10, 30}
assert window_ids == {0, 1, 4, 5}
print(sorted(review_ids))
```

The unusual second window makes recording 30 stand out. Once selected, both of its windows are marked. Use the recording IDs to queue the source audio and the window IDs to look up its stored vectors.

## How it works

1. `GoldSelector` sets up a Pixeltable result table keyed by `idx_vector`. It counts distinct eligible `idx` values when resolving the sample budget, then collects candidate vectors. It can divide the budget by label and process candidates in chunks when needed.
2. Greedy K-center computes pairwise distances. With no anchors, it starts at the vector whose distances to the candidate set have the largest L2 norm. After each pick, it updates each remaining vector's distance to the nearest center and chooses the one with the largest such distance. Later rounds can use vector rows from already-selected samples as anchors.
3. The tool returns positions in its input tensor. The selector maps those positions to sample IDs, marks every vector row of each winning sample, and can make another pass if several picks resolved to the same sample. Retrieval by `idx` gives distinct sample IDs.

## Choose the representation and the objective

Use a pretrained model through `GoldTorchEmbeddingTool` and `GoldDescriptor` to extract embeddings. Use `GoldTensorVectorizationTool` in the descriptor, or `GoldVectorizer` on stored embeddings, to turn model features into vectors. Point the selector's `vectorized_key` at the resulting column. The embedding layer and distance metric determine what looks redundant to the selector.

Use an integer `select_size` for a fixed budget or a float for a fraction of distinct samples, rounded up. Set `label_key` if you already have labels and want selection to account for them. A small budget can be insufficient to cover every label. Earlier assignments in the table count toward a repeated request for the same value and reserve those samples from later values. Reuse the table for the same run. Choose a fresh table for a different representation or selection recipe.

Greedy K-center emphasizes coverage. `GoldGreedyKernelPointsSelectionTool` aims to approximate a distribution in its chosen kernel space. For a large pool, chunking limits the per-chunk computation and makes coverage decisions within smaller pools. Reducing vector width also saves work, with the risk of erasing distinctions you need. [Coreset selection survey](https://arxiv.org/abs/2505.17799) · [Reducing the dimensionality of embeddings](GOLD_REDUCTION.md)

For monitoring, review the selected IDs to discover failure modes and collect examples for follow-up. Coverage selection changes the mixture you inspect, so draw a separate random audit sample from the deployment population when estimating deployment error rates. If the samples are already labeled and you need disjoint training and evaluation sets, use the same selection stage inside `GoldSplitter`. [Splitting labeled data with Goldener](GOLD_SPLIT.md)

Original post by [Yann Chéné](https://huggingface.co/Yann-CV). Sponsored by [Pixeltable](https://www.pixeltable.com/).
