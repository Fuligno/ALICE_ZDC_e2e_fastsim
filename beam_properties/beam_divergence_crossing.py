#!/usr/bin/env python3

from __future__ import annotations

import math
from pathlib import Path

PB208_Z = 82
PB208_A = 208
ATOMIC_MASS_UNIT_TEV = 0.00093149410242
OUTPUT_FILE = Path(__file__).with_suffix(".txt")

YEAR_DATA = {
    2018: {
        "beam_energy_ztev": 6.37,
        "beta_star_m": 0.5,
        "emittance_um": (1.3, 1.5),
        "external_crossing_urad": (137.0,),
        "internal_crossing_urad": (-77.0,),
        "total_half_crossing_urad": (60.0,),
        "source": "https://lpc.web.cern.ch/Run2/HIConfiguration2018.html",
    },
    2023: {
        "beam_energy_ztev": 6.8,
        "beta_star_m": 0.5,
        "emittance_um": (1.65,),
        "external_crossing_urad": (-170.0, 170.0),
        "internal_crossing_urad": (72.0, -72.0),
        "total_half_crossing_urad": (-98.0, 98.0),
        "source": "https://lpc.web.cern.ch/Run3/HIConfiguration2023.html",
    },
    2024: {
        "beam_energy_ztev": 6.8,
        "beta_star_m": 0.5,
        "emittance_um": (1.65,),
        "external_crossing_urad": (-150.0, 150.0),
        "internal_crossing_urad": (72.0, -72.0),
        "total_half_crossing_urad": (-78.0, 78.0),
        "source": "https://lpc.web.cern.ch/Run3/HIConfiguration2024.html",
    },
    2025: {
        "beam_energy_ztev": 6.8,
        "beta_star_m": 0.5,
        "emittance_um": (2.1, 2.4),
        "external_crossing_urad": (-150.0, 150.0),
        "internal_crossing_urad": (72.0, -72.0),
        "total_half_crossing_urad": (-78.0, 78.0),
        "source": "https://lpc.web.cern.ch/Run3/HIConfiguration2025.html",
    },
}


def beam_gamma(beam_energy_ztev: float, charge_number: int = PB208_Z, mass_number: int = PB208_A) -> float:
    energy_per_nucleon_tev = beam_energy_ztev * charge_number / mass_number
    return energy_per_nucleon_tev / ATOMIC_MASS_UNIT_TEV


def beam_divergence_urad(normalized_emittance_um: float, beta_star_m: float, gamma_rel: float) -> float:
    geometric_emittance = normalized_emittance_um * 1e-6 / gamma_rel
    return math.sqrt(geometric_emittance / beta_star_m) * 1e6


def format_values(values: tuple[float, ...], unit: str, precision: int = 1) -> str:
    if len(values) == 1:
        return f"{values[0]:.{precision}f} {unit}"
    ordered = tuple(sorted(values))
    return ", ".join(f"{value:.{precision}f} {unit}" for value in ordered)


def format_range(values: tuple[float, ...], unit: str, precision: int = 1) -> str:
    if len(values) == 1:
        return f"{values[0]:.{precision}f} {unit}"
    ordered = tuple(sorted(values))
    return f"{ordered[0]:.{precision}f}-{ordered[-1]:.{precision}f} {unit}"

def build_output_lines() -> list[str]:
    lines = list()

    for year, data in YEAR_DATA.items():
        gamma_rel = beam_gamma(data["beam_energy_ztev"])
        divergences = tuple(
            beam_divergence_urad(emittance_um, data["beta_star_m"], gamma_rel)
            for emittance_um in data["emittance_um"]
        )

        lines.append(f"{year}")
        lines.append(f"  beam energy: {data['beam_energy_ztev']:.2f} ZTeV")
        lines.append(f"  gamma: {gamma_rel:.1f}")
        lines.append(f"  beta*: {data['beta_star_m']:.2f} m")
        lines.append(f"  normalized emittance: {format_range(data['emittance_um'], 'um', precision=2)}")
        lines.append(f"  beam divergence: {format_range(divergences, 'urad')}")
        lines.append(f"  total half crossing: {format_values(data['total_half_crossing_urad'], 'urad')}")
        lines.append("")

    return lines


def main() -> None:
    # ALICE/IP2 heavy-ion beam properties
    # Divergence is computed as sqrt((epsilon_n / gamma) / beta*), with beta_rel ~= 1

    output_lines = build_output_lines()
    output_text = "\n".join(output_lines)

    print(output_text)
    OUTPUT_FILE.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()