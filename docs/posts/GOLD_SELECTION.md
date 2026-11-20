# Choose what to label with embedding-based selection

You can afford to label only a fraction of an unlabeled pool. When near-duplicates dominate, a random draw can spend much of the budget on examples that teach the model similar things. If you train a multilingual product search ranker, a queue dominated by common queries can hide regional product names and less common ways to describe the same item. Selecting for coverage can bring those queries to annotators when they form distinct neighborhoods.

Goldener compares vectors from a pretrained model and selects actual samples for review. The same workflow can build a monitoring queue for investigating unfamiliar inputs.

## Pick samples from an unlabeled pool

The example starts with six synthetic vectors in three neighborhoods. The IDs could be keys in your source dataset. The selector writes a named assignment to a local Pixeltable table, and the query returns sample IDs that you can send to an annotation queue.

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

Greedy K-center picks one sample from each of the three neighborhoods in this example. After choosing the first center, it repeatedly picks the candidate farthest from its *nearest* chosen center. Once a neighborhood has a center, taking another near-duplicate usually contributes less coverage. An isolated corrupt input can also look far away, so inspect the selected examples before labeling.

## Choose the representation and the objective

Use a pretrained model through `GoldTorchEmbeddingTool` and `GoldDescriptor` to extract embeddings. Use `GoldTensorVectorizationTool` in the descriptor, or `GoldVectorizer` on stored embeddings, to turn model features into vectors. Point the selector's `vectorized_key` at the resulting column. The embedding layer and distance metric determine what looks redundant to the selector. If you review long audio recordings for short sound events, use vectors for windows so a brief event can influence which recording you review. A whole-recording average can bury it.

The tool sees a 2D tensor with one vector per row and returns row positions. `GoldSelector` handles dataset or Pixeltable input and records the result by ID. `idx` identifies the sample you can label. `idx_vector` identifies a particular vector, such as one window in a recording. If a window is selected, its recording consumes one place in the budget and its other vectors are assigned to the same selection.

Use an integer `select_size` for a fixed budget or a float for a fraction of distinct samples, rounded up. Set `label_key` if you already have labels and want selection to account for them. A small budget can be insufficient to cover every label. Earlier assignments in the table count toward a repeated request for the same value and reserve those samples from later values. Reuse the table for the same run. Choose a fresh table for a different representation or selection recipe.

Greedy K-center emphasizes coverage. `GoldGreedyKernelPointsSelectionTool` aims to approximate a distribution in its chosen kernel space. For a large pool, chunking limits the per-chunk computation and makes coverage decisions within smaller pools. Reducing vector width also saves work, with the risk of erasing distinctions you need. [Coreset selection survey](https://arxiv.org/abs/2505.17799) · [Reducing the dimensionality of embeddings](GOLD_REDUCTION.md)

For monitoring, review the selected IDs to discover failure modes and collect examples for follow-up. Coverage selection changes the mixture you inspect, so draw a separate random audit sample from the deployment population when estimating deployment error rates. If the samples are already labeled and you need disjoint training and evaluation sets, use the same selection stage inside `GoldSplitter`. [Splitting labeled data with Goldener](GOLD_SPLIT.md)

Original post by [Yann Chéné](https://huggingface.co/Yann-CV). Sponsored by [Pixeltable](https://www.pixeltable.com/).
