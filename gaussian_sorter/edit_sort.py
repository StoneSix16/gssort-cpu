"""Select existing GSRT frame records and write a smaller GSRT file.

Frame selection is expressed using the stored frame_index values. The selected
records retain their original frame_index and sorted Gaussian index arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .gsrt import GSRTWriter, read_frame, read_header


def _expand_token(token: str) -> set[int]:
    token = token.strip()
    if not token:
        return set()
    if ":" in token:
        range_part, step_part = token.split(":", 1)
        step = int(step_part)
        if step <= 0:
            raise ValueError("Frame range step must be positive")
    else:
        range_part = token
        step = 1
    if "-" not in range_part:
        value = int(range_part)
        if value < 0:
            raise ValueError("Frame indices must be non-negative")
        return {value}
    start_text, end_text = range_part.split("-", 1)
    start = int(start_text)
    end = int(end_text)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"Invalid frame range: {token}")
    return set(range(start, end + 1, step))


def parse_frame_expression(expression: str) -> set[int]:
    """Parse values such as ``0,3,10-20:2`` into stored frame indices."""

    selected: set[int] = set()
    for token in expression.split(","):
        selected.update(_expand_token(token))
    if not selected:
        raise ValueError("Frame selection cannot be empty")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path; defaults to <input>.edited.sort",
    )
    parser.add_argument(
        "--frames",
        required=True,
        help="Stored frame indices, for example 0,3,10-20:2",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow editing an input file whose status is 0",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input.resolve()
    output_path = (args.output or Path(str(args.input) + ".edited.sort")).resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")

    header = read_header(input_path)
    if header.status != 1 and not args.allow_incomplete:
        raise ValueError(
            "Input GSRT file is incomplete (status=0); use --allow-incomplete to override"
        )
    selected_indices = parse_frame_expression(args.frames)

    selected_positions: list[int] = []
    selected_frame_indices: list[int] = []
    for position in range(header.frame_count):
        frame_index, _, _ = read_frame(input_path, position)
        if frame_index in selected_indices:
            selected_positions.append(position)
            selected_frame_indices.append(frame_index)
    if not selected_positions:
        raise ValueError("No stored frames matched --frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with GSRTWriter(
        output_path,
        gaussian_count=header.gaussian_count,
        frame_count=len(selected_positions),
        version=header.version,
    ) as writer:
        for position in selected_positions:
            frame_index, valid_count, sorted_indices = read_frame(input_path, position)
            writer.write_frame(frame_index, valid_count, sorted_indices)
        writer.finalize()

    print(f"input frames: {header.frame_count}")
    print(f"output frames: {len(selected_positions)}")
    print(f"gaussians per frame: {header.gaussian_count}")
    print(f"preserved frame indices: {', '.join(str(value) for value in selected_frame_indices)}")
    print(f"completed: {output_path}")


if __name__ == "__main__":
    main()
