"""Shared complex-tensor primitives for AC power-flow physics.

The functions intentionally preserve the original tensor dtype: float32
inputs produce complex64 tensors, while float64 inputs produce complex128.
They run on CUDA as the primary execution path and remain usable on CPU for
tests or analysis.
"""

from __future__ import annotations

import torch


def normalize_tap(tap: torch.Tensor) -> torch.Tensor:
    """Map MATPOWER's zero (no-transformer) tap convention to one."""
    return torch.where(tap == 0, torch.ones_like(tap), tap)


def branch_complex_powers(
    voltage: torch.Tensor,
    angle: torch.Tensor,
    resistance: torch.Tensor,
    reactance: torch.Tensor,
    tap: torch.Tensor,
    shift: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    total_susceptance: torch.Tensor | None = None,
    conductance_from: torch.Tensor | None = None,
    susceptance_from: torch.Tensor | None = None,
    conductance_to: torch.Tensor | None = None,
    susceptance_to: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complex powers injected into a branch at its two endpoints.

    Endpoint shunts are modelled as independent complex admittances.  Callers
    that only retain total charging susceptance can pass
    ``total_susceptance``; it is then split equally between the endpoints to
    preserve the legacy PFNet feature contract.
    """
    endpoint_terms = (
        conductance_from,
        susceptance_from,
        conductance_to,
        susceptance_to,
    )
    has_endpoint_terms = [term is not None for term in endpoint_terms]
    if any(has_endpoint_terms) and not all(has_endpoint_terms):
        raise ValueError("Endpoint branch shunts must be supplied together")

    if all(has_endpoint_terms):
        shunt_from = conductance_from + 1j * susceptance_from
        shunt_to = conductance_to + 1j * susceptance_to
    else:
        if total_susceptance is None:
            total_susceptance = torch.zeros_like(resistance)
        shunt_from = 1j * total_susceptance / 2
        shunt_to = 1j * total_susceptance / 2

    complex_voltage = voltage * torch.exp(1j * angle)
    transformer_tap = normalize_tap(tap) * torch.exp(1j * shift)
    series_admittance = 1 / (resistance + 1j * reactance)
    voltage_src = complex_voltage[src]
    voltage_dst = complex_voltage[dst]

    power_src = (
        (series_admittance + shunt_from).conj()
        * voltage_src.abs().square()
        / transformer_tap.abs().square()
        - series_admittance.conj()
        * voltage_src
        * voltage_dst.conj()
        / transformer_tap
    )
    power_dst = (
        (series_admittance + shunt_to).conj() * voltage_dst.abs().square()
        - series_admittance.conj()
        * voltage_dst
        * voltage_src.conj()
        / transformer_tap.conj()
    )
    return power_src, power_dst


def branch_power_injections(
    voltage: torch.Tensor,
    angle: torch.Tensor,
    resistance: torch.Tensor,
    reactance: torch.Tensor,
    tap: torch.Tensor,
    shift: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    **shunt_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate complex endpoint powers into real nodal P/Q injections."""
    power_src, power_dst = branch_complex_powers(
        voltage,
        angle,
        resistance,
        reactance,
        tap,
        shift,
        src,
        dst,
        **shunt_kwargs,
    )
    active = torch.zeros_like(voltage)
    reactive = torch.zeros_like(voltage)
    active.index_add_(0, src, power_src.real)
    active.index_add_(0, dst, power_dst.real)
    reactive.index_add_(0, src, power_src.imag)
    reactive.index_add_(0, dst, power_dst.imag)
    return active, reactive
