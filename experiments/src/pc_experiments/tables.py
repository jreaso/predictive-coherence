"""The values behind the paper's tables, written as CSV: snake_case columns, numbers at full
precision, and an empty cell where a value is missing."""

import csv
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from pc_experiments.outputs import table_path

Cell = str | int | float | None

_COLUMN = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class Table:
    name: str
    columns: Sequence[str]
    rows: Sequence[Sequence[Cell]]

    def __post_init__(self) -> None:
        for column in self.columns:
            if not _COLUMN.fullmatch(column):
                raise ValueError(f"{self.name}: column {column!r} is not ASCII snake_case")
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError(
                    f"{self.name}: a row of {len(row)} cells under {len(self.columns)}"
                )


def cell(value: Cell) -> str:
    """A value as written: empty for `None` or NaN, a float to full precision."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Real):
        number = float(value)
        return "" if math.isnan(number) else repr(number)
    raise TypeError(f"a cell of type {type(value).__name__}")


def write_tables(experiment: str, tables: Sequence[Table]) -> None:
    """Write each table to `experiments/tables/<experiment>_<name>.csv`."""
    for table in tables:
        path = table_path(experiment, table.name)
        with path.open("w", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(table.columns)
            writer.writerows([cell(value) for value in row] for row in table.rows)
        print(f"  {path.relative_to(path.parents[1])}", flush=True)
