"""Small, deterministic air-corridor prototype.

The prototype is intentionally independent of the existing project code so it
can be run with the Python standard library only.  It demonstrates:
  * layer feasibility from obstacle height and clearance;
  * lane-change action masking;
  * priority-based intersection reservation;
  * sensitivity to the UAV-UAV safety distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Layer:
    name: str
    z_low: float
    z_high: float

    @property
    def center(self) -> float:
        return 0.5 * (self.z_low + self.z_high)


@dataclass(frozen=True)
class Corridor:
    name: str
    length_m: float
    max_obstacle_height_m: float


@dataclass(frozen=True)
class Request:
    uav_id: int
    eta_s: float
    urgency: int
    emergency: int
    layer: str
    lane: int
    speed_mps: float


@dataclass(frozen=True)
class Reservation:
    uav_id: int
    t_in_s: float
    t_out_s: float
    layer: str
    lane: int


LAYERS = (
    Layer("L1", 18.0, 24.0),
    Layer("L2", 30.0, 36.0),
    Layer("L3", 42.0, 48.0),
)


def choose_layer(
    corridor: Corridor,
    layers: Iterable[Layer],
    obstacle_clearance_m: float = 5.0,
    z_reg_min_m: float = 10.0,
    z_reg_max_m: float = 50.0,
) -> Tuple[Layer, List[Tuple[str, str]]]:
    """Choose the least-cost feasible example layer.

    The regulatory bounds here are explicitly scenario bounds, not a legal
    conclusion.  L2 wins in the toy example because L1 cannot clear a 25 m
    obstacle with 5 m clearance.
    """
    candidates: List[Tuple[float, Layer]] = []
    rejected: List[Tuple[str, str]] = []
    for layer in layers:
        if layer.z_low < z_reg_min_m or layer.z_high > z_reg_max_m:
            rejected.append((layer.name, "outside scenario bounds"))
            continue
        if layer.z_low < corridor.max_obstacle_height_m + obstacle_clearance_m:
            rejected.append((layer.name, "insufficient obstacle clearance"))
            continue
        # Example cost: vertical movement plus a small height-dependent risk.
        vertical_energy_proxy = 2.0 * abs(layer.center - 20.0)
        risk_proxy = 0.02 * layer.center
        candidates.append((vertical_energy_proxy + risk_proxy, layer))
    if not candidates:
        raise RuntimeError(f"No feasible layer for corridor {corridor.name}")
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[0][1], rejected


def lane_change_allowed(
    *,
    arc_position_m: float,
    zone: Tuple[float, float],
    in_conflict_zone: bool,
    target_lane_available: bool,
    predicted_gap_m: float,
    d_safe_uav_m: float,
    communication_ok: bool,
) -> Tuple[bool, str]:
    """Apply the hard action mask for one lane-change request."""
    if in_conflict_zone:
        return False, "inside conflict zone"
    if not (zone[0] <= arc_position_m <= zone[1]):
        return False, "outside lane-change zone"
    if not target_lane_available:
        return False, "target lane reserved"
    if predicted_gap_m < d_safe_uav_m:
        return False, "predicted gap below d_safe_uav"
    if not communication_ok:
        return False, "communication degraded"
    return True, "accepted"


def conflict_separation_s(
    *,
    conflict_length_m: float,
    v_min_mps: float,
    d_safe_uav_m: float,
    tau_comm_s: float = 0.2,
    tau_ctrl_s: float = 0.1,
    tau_margin_s: float = 0.5,
) -> float:
    """Conservative intersection time separation in seconds."""
    occupancy_s = conflict_length_m / v_min_mps
    sensing_control_s = 2.0 * tau_comm_s + tau_ctrl_s + tau_margin_s
    following_gap_s = d_safe_uav_m / v_min_mps + tau_comm_s + tau_ctrl_s
    return max(occupancy_s + sensing_control_s, following_gap_s)


def reserve_intersection(
    requests: Iterable[Request],
    *,
    conflict_length_m: float,
    v_min_mps: float,
    d_safe_uav_m: float,
) -> List[Reservation]:
    """Priority reservation for one shared conflict zone."""
    ordered = sorted(
        requests,
        key=lambda r: (-r.emergency, -r.urgency, r.eta_s, r.uav_id),
    )
    separation_s = conflict_separation_s(
        conflict_length_m=conflict_length_m,
        v_min_mps=v_min_mps,
        d_safe_uav_m=d_safe_uav_m,
    )
    reservations: List[Reservation] = []
    for request in ordered:
        t_in = request.eta_s
        for prior in reservations:
            same_layer = prior.layer == request.layer
            if same_layer and t_in < prior.t_in_s + separation_s:
                t_in = prior.t_in_s + separation_s
        t_out = t_in + conflict_length_m / request.speed_mps
        reservations.append(
            Reservation(request.uav_id, t_in, t_out, request.layer, request.lane)
        )
    return sorted(reservations, key=lambda r: r.t_in_s)


def min_time_gap(reservations: Iterable[Reservation]) -> Optional[float]:
    ordered = sorted(reservations, key=lambda r: r.t_in_s)
    if len(ordered) < 2:
        return None
    return min(b.t_in_s - a.t_in_s for a, b in zip(ordered, ordered[1:]))


def run_once(d_safe_uav_m: float) -> Tuple[float, float, List[Reservation]]:
    requests = [
        Request(1, eta_s=0.0, urgency=2, emergency=0, layer="L2", lane=0, speed_mps=5.0),
        Request(2, eta_s=0.0, urgency=3, emergency=0, layer="L2", lane=1, speed_mps=5.0),
        Request(3, eta_s=0.0, urgency=1, emergency=1, layer="L2", lane=0, speed_mps=5.0),
        Request(4, eta_s=0.0, urgency=1, emergency=0, layer="L2", lane=1, speed_mps=5.0),
    ]
    reservations = reserve_intersection(
        requests,
        # A short teaching intersection makes the following-gap sensitivity visible.
        conflict_length_m=5.0,
        v_min_mps=5.0,
        d_safe_uav_m=d_safe_uav_m,
    )
    separation_s = conflict_separation_s(
        conflict_length_m=5.0,
        v_min_mps=5.0,
        d_safe_uav_m=d_safe_uav_m,
    )
    gap_s = min_time_gap(reservations)
    assert gap_s is not None and gap_s + 1e-12 >= separation_s
    return separation_s, gap_s, reservations


def dimension_checks() -> None:
    """Numerical checks for the principal unit identities."""
    dt_s = 0.1
    velocity_mps = 15.0
    acceleration_mps2 = 5.0
    assert abs(velocity_mps * dt_s - 1.5) < 1e-12  # m/s * s = m
    assert abs(acceleration_mps2 * dt_s**2 - 0.05) < 1e-12  # m/s^2 * s^2 = m
    assert abs(20.0 / 5.0 - 4.0) < 1e-12  # m / (m/s) = s
    energy_j = 1.5 * 9.81 * 13.0 / 0.8  # kg*m/s^2*m = J
    assert energy_j > 0.0


def main() -> None:
    dimension_checks()
    corridor = Corridor("J_main", length_m=100.0, max_obstacle_height_m=25.0)
    selected, rejected = choose_layer(corridor, LAYERS)
    print("[layer-selection]")
    print(f"corridor={corridor.name}, obstacle_height=25.0 m, selected={selected.name} ({selected.z_low:.0f}-{selected.z_high:.0f} m)")
    print(f"rejected={rejected}")

    accepted, reason = lane_change_allowed(
        arc_position_m=20.0,
        zone=(15.0, 35.0),
        in_conflict_zone=False,
        target_lane_available=True,
        predicted_gap_m=12.0,
        d_safe_uav_m=10.0,
        communication_ok=True,
    )
    rejected_change, rejected_reason = lane_change_allowed(
        arc_position_m=45.0,
        zone=(15.0, 35.0),
        in_conflict_zone=True,
        target_lane_available=True,
        predicted_gap_m=20.0,
        d_safe_uav_m=10.0,
        communication_ok=True,
    )
    print("[lane-change]")
    print(f"upstream_request={accepted} ({reason}); conflict_zone_request={rejected_change} ({rejected_reason})")

    print("[reservation]")
    for safe_distance in (5.0, 10.0, 15.0):
        separation_s, gap_s, reservations = run_once(safe_distance)
        order = " -> ".join(f"UAV{r.uav_id}@{r.t_in_s:.1f}s" for r in reservations)
        print(f"d_safe_uav={safe_distance:>4.1f} m | T_sep={separation_s:.2f} s | min_gap={gap_s:.2f} s | {order}")

    print("[checks] dimensions=PASS, hard-interval-separation=PASS")


if __name__ == "__main__":
    main()
