"""Tables written for loading."""

import csv
import math

import pytest

from pc_experiments import tables
from pc_experiments.tables import Table, cell, write_tables


def test_cells_are_full_precision_and_empty_when_missing():
    assert cell(0.1 + 0.2) == "0.30000000000000004"
    assert float(cell(1 / 3)) == 1 / 3
    assert cell(2.58e-28) == "2.58e-28"
    assert cell(7) == "7"
    assert cell(True) == "1"
    assert cell(None) == ""
    assert cell(math.nan) == ""
    assert cell("sparse_gp") == "sparse_gp"


def test_columns_must_be_snake_case_and_rows_complete():
    with pytest.raises(ValueError, match="snake_case"):
        Table("t", ["Median Ψ"], [])
    with pytest.raises(ValueError, match="snake_case"):
        Table("t", ["spread over splits"], [])
    with pytest.raises(ValueError, match="a row of 1"):
        Table("t", ["model", "psi"], [["gp"]])


def test_files_are_written_under_the_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tables, "table_path", lambda experiment, name: tmp_path / f"{experiment}_{name}.csv"
    )
    write_tables("demo", [Table("results", ["model", "psi", "psi_se"], [["gp", 0.5, None]])])
    with (tmp_path / "demo_results.csv").open() as file:
        assert list(csv.reader(file)) == [["model", "psi", "psi_se"], ["gp", "0.5", ""]]
