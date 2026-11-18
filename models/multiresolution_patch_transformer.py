"""Options for multiscale history patches."""

from dataclasses import dataclass

from models.full_future_common import FullFutureOptions


@dataclass(frozen=True)
class PatchFutureOptions(FullFutureOptions):
    patch_lengths: tuple[int, ...] = (4, 8, 16)

    def __post_init__(self):
        super().__post_init__()
        values = tuple(self.patch_lengths)
        if not values or any(type(v) is not int or v < 1 for v in values):
            raise ValueError('patch_lengths must be a nonempty sequence of positive integers.')
        object.__setattr__(self, "patch_lengths", values)
