from dataclasses import dataclass

from goldener.utils import get_ratio_list_sum


@dataclass
class GoldSet:
    name: str
    ratio: float

    def __post_init__(self):
        if not (0 < self.ratio < 1):
            raise ValueError("Ratio must be between 0 and 1.")


class GoldSplitter:
    def __init__(
        self,
        sets: list[GoldSet],
    ) -> None:
        ratios_sum = get_ratio_list_sum([s.ratio for s in sets])
        ratios_names = [s.name for s in sets]
        if len(ratios_names) != len(set(ratios_names)):
            raise ValueError(f"Set names must be unique, got {ratios_names}")

        if ratios_sum != 1.0:
            sets.append(GoldSet(name="not assigned", ratio=1.0 - ratios_sum))
