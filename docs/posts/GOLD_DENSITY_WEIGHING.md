# Reweight training samples by embedding density

Two examples can carry the same label even when one reflects a much less common case. In a table of delayed shipments, most delayed examples might be from busy routes. A model still needs to learn about delays on less traveled routes. If those routes form sparse neighborhoods in the feature space, local distances offer a way to give them more influence without changing the label.

`GoldDensityWeigher` scores each vector by how far it sits from nearby vectors. Use those scores to draw some samples more often or to give them more weight in the training loss.

## Weigh samples and draw a training batch

Here, eight samples share a label. Six sit close together in embedding space, and two sit apart. The k-nearest-neighbor tool scores their vectors, and the weigher looks up the resulting sample weights by ID. The sampler needs weights in the same order as the training dataset.

```python
from uuid import uuid4

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from goldener import GoldDensityWeigher, GoldKnnDensityWeighingTool

vectors = torch.tensor(
    [
        [0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1],
        [0.2, 0.0], [0.0, 0.2], [3.0, 3.0], [5.0, 3.0],
    ],
)
sample_ids = [101, 205, 309, 412, 518, 623, 731, 845]
training_data = [
    {"idx": sample_id, "idx_vector": pos, "vectorized": vector, "label": "same_label"}
    for pos, (sample_id, vector) in enumerate(zip(sample_ids, vectors))
]

weigher = GoldDensityWeigher(
    table_path=f"training_weights_{uuid4().hex}",
    weigher=GoldKnnDensityWeighingTool(k=2),
    label_key="label",
)
table = weigher.weigh_in_table(training_data)
sample_weights = weigher.get_weights_per_idx(table)
weights_in_order = [sample_weights[row["idx"]] for row in training_data]

assert min(weights_in_order[-2:]) > max(weights_in_order[:6])
sampler = WeightedRandomSampler(
    weights_in_order,
    num_samples=len(training_data),
    replacement=True,
    generator=torch.Generator().manual_seed(42),
)
loader = DataLoader(training_data, batch_size=4, sampler=sampler)
for batch in loader:
    print(batch["idx"].tolist())
```

The two isolated samples have larger weights, so each has a higher probability per draw. Sampling with replacement can repeat some examples and leave others undrawn in an epoch. If you want to keep the existing sampler, use the weights in the loss instead. Multiply the unreduced per-sample losses by their weights and divide their sum by the total positive weight in the batch. Using both a weighted sampler and a weighted loss compounds the emphasis.

## Decide which neighbors count

`GoldKnnDensityWeighingTool` averages the distances to the `k` nearest eligible neighbors, floors the mean at a small positive value, then raises it to `power`. The default square root softens differences in distance. Euclidean distance can reflect both magnitude and direction. Cosine distance compares directions, which is useful when vector magnitude carries little meaning. Large scores signal sparsity in the chosen representation. Inspect highly weighted examples before you use them. In a camera-trap image dataset, a sparse neighborhood might contain useful night sightings or blank frames from a camera fault. In the second case, remove the faulty inputs or change the representation before increasing their influence.

Set `label_key` to look for neighbors within the same class, as above, or leave it unset for a shared population. Within-class scoring can give different weights to samples of the same label. Class balance still depends on the scores and training policy. [Cui et al. on reweighting by class frequency](https://arxiv.org/abs/1901.05555)

Use one vector per sample as shown, or keep several vectors for regions or views. The result table stores a raw score per `idx_vector`. `get_weights_per_idx` averages those scores by sample `idx` and divides by the largest returned sample score by default. This gives one weight per item for training. With several vectors per sample, set `exclude_same_sample=True` if multiple views of the same item should never count as neighbors. Each query then needs `k` eligible vectors from other samples and at least `k + 1` reference rows in its population. By default, the tool drops just one nearest reference and requires at least `k + 2` vectors per population.

## Balance memory use and neighborhood scope

The weigher accepts vectors in a dataset or Pixeltable table. If you store model embeddings with `GoldDescriptor`, vectorize them as needed and configure the weigher's `vectorized_key` to match. Set `weighing_batch_size` to limit how many queries are scored at once against the same reference population. Exact search still does pairwise work that grows quadratically with population size. Setting `chunk` reduces each query's reference population to its own chunk. The `k` requirement applies inside every scored chunk or label group. Saved scores are reused on resume, while pending vectors are scored against other pending vectors. Use a fresh result table when you need one consistent calculation over the full population after an interruption or a change in settings.

To control which modes appear together in each training batch, use a cluster-balanced batch sampler. [Building batches from embedding clusters](GOLD_BATCHER.md)
