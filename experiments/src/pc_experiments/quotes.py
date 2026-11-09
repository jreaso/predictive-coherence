"""Printing the numbers the paper quotes, under the part of the paper they appear in."""


def section(title: str) -> None:
    print(f"\n{title}", flush=True)


def quote(label: str, value: str) -> None:
    print(f"  {label}: {value}", flush=True)
