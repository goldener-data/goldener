# Split labeled data with semantic coverage in mind

## TL;DR

Use `GoldSplitter` to assign samples to disjoint sets with semantic coverage in mind. Earlier sets get first choice, and the last receives the remainder. Establish any grouping or time boundaries before splitting.

## Table of contents

- [Build a diagnostic validation set for video](#build-a-diagnostic-validation-set-for-video)
- [Decide which set gets first choice](#decide-which-set-gets-first-choice)
- [Hold back future transactions before splitting history](#hold-back-future-transactions-before-splitting-history)
- [How it works](#how-it-works)
- [Design the split around the evaluation question](#design-the-split-around-the-evaluation-question)

A train and validation split can match label counts while missing variation within a label. For a video activity classifier, a small validation set could contain plenty of clips of the common activities, filmed mostly from a single viewpoint. Once you have set aside independent recordings for validation, embedding-based selection can help you choose clips that cover several viewpoints from those recordings.

`GoldSplitter` uses a selection algorithm to fill named sets in the order you supply. Give the first set a coverage objective, then decide whether the remaining data is suitable for the role you assign it.

## Build a diagnostic validation set for video

Here, nine independent video samples already have an embedding vector each. The vectors form three neighborhoods within the same class. We choose three validation samples first so that a small diagnostic set can cover all three modes. The training set gets the six remaining samples.

```python
from uuid import uuid4

import torch
from torch.utils.data import Subset

from goldener import GoldGreedyKCenterSelectionTool, GoldSelector, GoldSet, GoldSplitter

vectors = torch.tensor(
    [
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
        [4.0, 0.0], [4.1, 0.0], [4.0, 0.1],
        [0.0, 5.0], [0.1, 5.0], [0.0, 5.1],
    ],
)
rows = [
    {"idx": pos, "idx_vector": pos, "vectorized": vector, "label": "same_label"}
    for pos, vector in enumerate(vectors)
]
dataset = Subset(rows, range(len(rows)))

selector = GoldSelector(
    table_path=f"split_demo_{uuid4().hex}",
    selection_tool=GoldGreedyKCenterSelectionTool(device="cpu"),
    label_key="label",
)
splitter = GoldSplitter(
    sets=[GoldSet("validation", 3), GoldSet("train", 6)],
    selector=selector,
)
table = splitter.split_in_table(dataset)
splits = splitter.get_split_indices(table)

validation = Subset(dataset, sorted(splits["validation"]))
training = Subset(dataset, sorted(splits["train"]))
assert splits["validation"].isdisjoint(splits["train"])
assert (len(validation), len(training)) == (3, 6)
assert {pos // 3 for pos in splits["validation"]} == {0, 1, 2}
print(len(validation), len(training))  # 3 6
```

![The nine clips grouped into three embedding neighborhoods. Each neighborhood contributes one clip to validation and two to training.](figures/validation-neighborhoods.png)

`get_split_indices` groups the stored assignments by set name and returns sets of sample IDs. This example makes those IDs the dataset positions so they work directly in `Subset`. For catalog IDs, look up their dataset positions first.

## Decide which set gets first choice

Each `GoldSet` supplies a unique name and a target size. Use counts that sum to the number of distinct samples or fractions that sum to one. The splitter fills the sets in the order you give them and assigns the remainder to the last set. For earlier sets, it rounds fractions down with a minimum of one sample per set. Leave at least one sample for the final set. Fractions passed directly to `GoldSelector` round up.

Put validation first to prioritize its coverage, as in the example. Put training first when coverage matters most for training. Inspect each set's label and mode coverage, especially for small groups.

With `label_key`, the selector can raise an error if a set's initial allocation leaves a label with zero samples. Overlapping labels can also push a set above its target and leave fewer samples for the final set. Check the realized counts when sizes matter. [How the selector allocates its budget](GOLD_SELECTION.md#set-the-sample-budget)

## Hold back future transactions before splitting history

If a tabular model will score future transactions, put a date boundary around the test set before choosing any training examples. The synthetic dataset uses six historical rows and two future rows. Goldener builds a training and development split only within the historical window.

```python
from uuid import uuid4

import torch
from torch.utils.data import Subset

from goldener import GoldGreedyKCenterSelectionTool, GoldSelector, GoldSet, GoldSplitter

features = torch.tensor(
    [
        [0.0, 0.0], [0.1, 0.0], [2.0, 1.0], [2.1, 1.1],
        [4.0, 0.0], [4.1, 0.1], [4.5, 1.0], [5.0, 1.2],
    ],
)
rows = [
    {"idx": pos, "vectorized": vector, "day": pos + 1}
    for pos, vector in enumerate(features)
]
past_ids = [row["idx"] for row in rows if row["day"] <= 6]
future_ids = [row["idx"] for row in rows if row["day"] > 6]
past = Subset(rows, past_ids)
future_test = Subset(rows, future_ids)

selector = GoldSelector(
    table_path=f"transaction_split_{uuid4().hex}",
    selection_tool=GoldGreedyKCenterSelectionTool(device="cpu"),
)
splitter = GoldSplitter(
    sets=[GoldSet("train", 0.75), GoldSet("development", 0.25)],
    selector=selector,
)
splits = splitter.get_split_indices(splitter.split_in_table(past))

assert len(splits["train"]) == 4
assert splits["train"] | splits["development"] == set(past_ids)
assert (splits["train"] | splits["development"]).isdisjoint(future_ids)
print(len(splits["train"]), len(splits["development"]), len(future_test))  # 4 2 2
```

Training gets first choice within the past window. Development receives the remaining historical rows, and the future test set stays outside selection. Use fixed features for selection, as above. Fit the model's learned preprocessing on the chosen training rows before using it to score future rows.

## How it works

1. `GoldSplitter` takes vectors directly or runs the supplied `GoldDescriptor` and `GoldVectorizer` first. Supply a `GoldClusterizer` and `n_clusters > 1` to allocate selection budgets across embedding clusters. Match the output column of each stage to the next stage's input.
2. The splitter counts distinct sample IDs in the selector's table and resolves earlier set sizes against that original count. Each active selection writes its set name to the rows belonging to the chosen IDs, excluding those IDs from subsequent sets. A sample with several vector rows, such as clips represented by several views, keeps one set assignment. The final set is an update of all still-unassigned rows.
3. The selector persists assignments in Pixeltable. `get_split_indices` groups them into one set of IDs per name. Reuse the table to resume the same split. Start a new table when the order or selection criteria change.

[Selecting an embedding-based subset](GOLD_SELECTION.md) · [Clustering embeddings](GOLD_CLUSTERIZER.md)

## Design the split around the evaluation question

A coverage-oriented validation set can expose failures on familiar modes. Keep all clips from a held-out video recording out of training, including clips you skip when choosing a diagnostic review subset. With tabular transaction data, use a later holdout if the question is how a model handles changes in customer behavior after deployment. Keep the date boundary intact when selecting training examples. The splitter tracks samples by ID, so grouping related records and enforcing a time boundary happen outside this workflow.

Use a holdout that reflects the population you want to score when the result needs to estimate deployment performance. The diagnostic validation set above is intentionally chosen for variation. Joseph and Vakayil study another approach to geometry-based splitting, using support points to reflect the full dataset's distribution. [SPlit: An optimal method for data splitting](https://doi.org/10.1080/00401706.2021.1921037)

Sponsored by [Pixeltable](https://www.pixeltable.com/).
