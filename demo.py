"""Pauli product quantum computation, end to end.

  1. Litinski's Fig. 4 circuit, as Pauli product rotations.
  2. Compiled to pi/8 rotations followed by Pauli product measurements.
  3. Run on Clifft with one magic state per rotation, from two sources:
     bare noisy T states, and H6-distilled states built by LightStim.

Usage: python demo.py [--shots N] [--p CIRCUIT_NOISE] [--out results.png]
"""
import argparse

import numpy as np

from ppqc import (BareMagic, H6Magic, circuit_distribution, compile_to_ppm, litinski_fig4,
                  ppm_distribution, run, t_layers, total_variation)

P_T = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def plot(rows, floor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_t = [r[0] for r in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_xscale("log")
        ax.set_xticks(p_t, [f"{p:g}" for p in p_t])
        ax.minorticks_off()
        ax.set_xlabel("depolarizing error per T gate", color=MUTED)

    series = [("bare T states", BLUE, [r[1] for r in rows]),
              ("H6-distilled", ORANGE, [r[2] for r in rows])]
    for label, color, values in series:
        left.plot(p_t, values, color=color, linewidth=2, marker="o", markersize=6,
                  markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        left.annotate(label, (p_t[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                      color=INK, va="center", fontsize=9)
    left.axhline(floor, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    left.annotate("sampling floor", (p_t[0], floor), xytext=(0, 4), textcoords="offset points",
                  color=MUTED, fontsize=8)
    left.set_yscale("log")
    left.set_title("Output error of the Fig. 4 circuit", color=INK, loc="left", fontsize=11)
    left.set_ylabel("total variation distance from ideal", color=MUTED)
    left.legend(frameon=False, labelcolor=INK, loc="upper left")
    left.set_xlim(p_t[0] * 0.9, p_t[-1] * 1.7)

    right.plot(p_t, [r[3] for r in rows], color=ORANGE, linewidth=2, marker="o", markersize=6,
               markeredgecolor=SURFACE, markeredgewidth=1.5)
    right.set_ylim(0, 1)
    right.set_title("H6-distilled: runs accepted", color=INK, loc="left", fontsize=11)
    right.set_ylabel("fraction of runs passing every H6 check", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shots", type=int, default=2_000_000, help="shots per branch")
    parser.add_argument("--p", type=float, default=0.0,
                        help="LightStim circuit-level noise on the H6 Clifford operations")
    parser.add_argument("--out", default="results.png")
    args = parser.parse_args()

    n = 4
    circuit = litinski_fig4()
    program = compile_to_ppm(circuit, n)
    ideal = circuit_distribution(circuit, n)

    print("Litinski Fig. 4 circuit:", len(circuit), "Pauli product rotations,",
          sum(op.angle.denominator == 8 for op in circuit), "of them pi/8")
    print("\npi/8 rotations after commuting the Cliffords out:")
    for k, layer in enumerate(t_layers(program.rotations), 1):
        print(f"  layer {k}:", "  ".join(str(p) for p in layer))
    print("Pauli product measurements (replacing Z on q1..q4):")
    print("  " + "  ".join(str(p) for p in program.measurements))
    print(f"\nexact check, compiled program vs original circuit: "
          f"TV = {total_variation(ppm_distribution(program), ideal):.1e}")

    branches = 2 ** len(program.rotations)
    print(f"\nsampling {branches} Clifford-correction branches x {args.shots:,} shots on Clifft")
    floor = total_variation(run(program, lambda: BareMagic(n), args.shots).distribution, ideal)
    print(f"noiseless run, TV from ideal (sampling floor): {floor:.5f}\n")
    print(f"{'p_T':>6}  {'bare TV':>9}  {'H6 TV':>9}  {'H6 accepted':>11}")
    rows = []
    for p_t in P_T:
        bare = run(program, lambda: BareMagic(n, p_t=p_t), args.shots)
        h6 = run(program, lambda: H6Magic(n, p_t=p_t, p=args.p), args.shots)
        rows.append((p_t, total_variation(bare.distribution, ideal),
                     total_variation(h6.distribution, ideal), h6.acceptance))
        print(f"{p_t:>6g}  {rows[-1][1]:>9.5f}  {rows[-1][2]:>9.5f}  {rows[-1][3]:>11.3f}")
    plot(rows, floor, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
