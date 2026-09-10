"""Run the dense (λ, φ) FIM studies for pitch 80 nm and 300 nm.

    python3 run_fim_dense.py --layout-only
    python3 run_fim_dense.py
    python3 run_fim_dense.py --pitch 80
    python3 run_fim_dense.py --pitch 300 --layout-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import load_config
from fim_study import run_fim_study
from recipe import expand_measurement_conditions

HERE = Path(__file__).resolve().parent
JOBS = {
    "80": HERE / "config_fim_dense_p80.yaml",
    "300": HERE / "config_fim_dense_p300.yaml",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense (λ, φ) FIM for pitch 80 and 300 nm")
    parser.add_argument(
        "--pitch",
        choices=("80", "300", "both"),
        default="both",
        help="Which pitch group to run (default both)",
    )
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="Print propagating / mask tables; do not call S4",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def run_one(pitch: str, *, layout_only: bool, plot: bool) -> None:
    cfg_path = JOBS[pitch]
    cfg = load_config(cfg_path)
    n_cond = len(expand_measurement_conditions(cfg))
    n_fwd = n_cond * (1 + len(cfg.inverse.param_names))
    out_dir = (HERE / cfg.paths.output_dir).resolve()
    print("=" * 72)
    print(
        f"pitch={cfg.structure.pitch_nm:g} nm  CD={cfg.structure.cd_nm:g} nm  "
        f"depth={cfg.structure.depth_nm:g} nm  NG={cfg.optical.NG}"
    )
    print(f"config={cfg_path.name}  conditions={n_cond}  S4 for J={n_fwd}")
    print(f"output={out_dir}")
    print("=" * 72)
    run_fim_study(cfg, out_dir, plot=plot, layout_only=layout_only)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pitches = ("80", "300") if args.pitch == "both" else (args.pitch,)
    for pitch in pitches:
        run_one(pitch, layout_only=args.layout_only, plot=not args.no_plot)


if __name__ == "__main__":
    main()
