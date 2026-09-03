"""
hydraulics.py
=============
The physics heart of the digital twin.

Given a storm and a drainage network, this works out - minute by minute - how
much water lands on the city, how much each pipe can carry away, and where the
water that has nowhere to go ends up ponding on the street.

Two standard civil-engineering methods do the heavy lifting:

  1. THE RATIONAL METHOD turns rainfall into runoff.
         Q = C * i * A / 360
     Q is flow in m3/s, C is the runoff coefficient (what fraction of the rain
     runs off rather than soaking in), i is rainfall intensity in mm/hr and A
     is the catchment area in hectares.

  2. MANNING'S EQUATION says how much a pipe can carry (see network.py).

On top of those we add a simple continuity (mass balance) step: water that
arrives at a junction but cannot fit into the outgoing pipe has to go
somewhere, and that somewhere is the street. That ponded water is exactly what
a flood is, and it is what we predict.

This is a simplification of what a full solver like EPA SWMM does. It is
deliberate: it runs in milliseconds, it is transparent enough to explain to a
judge in one minute, and it captures the behaviour that actually matters -
where the network runs out of capacity first.
"""

import math
from dataclasses import dataclass, field, asdict

import numpy as np

# Timestep for the simulation, in minutes. Five minutes is the standard
# resolution for urban drainage - the systems respond that quickly.
DT_MIN = 5.0
DT_SEC = DT_MIN * 60.0

# How long we keep simulating after the rain stops, so we can watch the
# network drain down and see how long streets stay flooded.
RECESSION_MIN = 90.0

# Depth thresholds in metres. These are the numbers safety guidance uses.
DEPTH_NUISANCE = 0.10    # standing water; drainage has failed here
DEPTH_PEDESTRIAN = 0.15  # unsafe to wade, especially for children
DEPTH_VEHICLE = 0.30     # a normal car will stall or start to float

# --- Overland ("major system") flow -----------------------------------------
# When a drain is overwhelmed, water does not pile up forever - it spills out
# of the manhole and runs down the street to the next low point. Drainage
# engineers call the pipes the MINOR system and the streets the MAJOR system,
# and modelling both together is called dual drainage. Ignoring the street
# half is the single biggest mistake a simple flood model can make.
#
# We treat the flooded street as a wide, shallow channel and apply Manning's
# equation to it - the same formula we used for the pipes, just with a
# different shape and a rougher surface.
STREET_WIDTH_M = 8.0      # effective width water sheets across
STREET_ROUGHNESS = 0.030  # Manning n for a road with kerbs, cars and debris
KERB_DEPTH_M = 0.03       # water fills the gutter before it sheets across
MAX_DEPTH_M = 3.0         # sanity cap so a runaway number cannot break the UI

# Beyond about a quarter of a metre, extra depth stops making the water run
# downhill faster - it just spreads sideways into yards, shops and buildings.
# Without this cap the Manning formula would predict a street carrying tens of
# cubic metres a second, which is a river, not a road.
OVERLAND_MAX_HEAD_M = 0.25


@dataclass
class Storm:
    """One rainfall scenario - the thing the user moves sliders to change."""

    peak_intensity_mmhr: float = 60.0   # peak rainfall rate
    duration_min: float = 60.0          # how long it rains
    antecedent_saturation: float = 0.3  # 0 = bone dry ground, 1 = already soaked
    blockage_factor: float = 0.15       # 0 = clean pipes, 1 = completely blocked

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class NodeResult:
    """What happened at one junction during one storm."""

    node_id: str
    peak_depth_m: float = 0.0
    peak_surcharge_ratio: float = 0.0   # peak inflow / pipe capacity
    flood_volume_m3: float = 0.0        # total water that spilled onto the street
    minutes_flooded: float = 0.0        # how long depth stayed above nuisance level
    time_to_flood_min: float = -1.0     # how much warning we would have had
    peak_inflow_m3s: float = 0.0


def rainfall_hyetograph(storm: Storm, total_steps: int) -> np.ndarray:
    """
    Build the storm profile: rainfall intensity at each timestep.

    Real storms are not flat. They build, peak, then tail off. We use a
    standard single-peak design storm with the peak at 40% of the way through,
    which is the shape most design guidance assumes.

    Returns an array of intensities in mm/hr, one per timestep.
    """
    rain_steps = max(1, int(math.ceil(storm.duration_min / DT_MIN)))
    profile = np.zeros(total_steps)

    # Position of each rainy timestep along the storm, from 0 to 1.
    t = (np.arange(rain_steps) + 0.5) / rain_steps
    peak_position = 0.40

    shape = np.where(
        t <= peak_position,
        t / peak_position,                        # rising limb
        (1.0 - t) / (1.0 - peak_position),        # falling limb
    )
    shape = np.clip(shape, 0.10, 1.0)             # never quite zero while raining

    profile[:rain_steps] = shape * storm.peak_intensity_mmhr
    return profile


class HydraulicModel:
    """
    Wraps a drainage network and runs storms through it.

    Build it once, then call simulate() as many times as you like - all the
    expensive setup (sorting the network, building lookup arrays) happens in
    the constructor.
    """

    def __init__(self, network: dict):
        self.network = network
        self.nodes = network["nodes"]
        self.node_ids = [n["id"] for n in self.nodes]
        self.index_of = {node_id: i for i, node_id in enumerate(self.node_ids)}
        self.n = len(self.nodes)

        # Pull the per-node numbers we need into flat arrays. Working with
        # arrays instead of dictionaries makes the inner loop far faster,
        # which matters because we run thousands of storms to train the model.
        self.area_ha = np.array([n["area_ha"] for n in self.nodes])
        self.runoff_coeff = np.array([n["runoff_coeff"] for n in self.nodes])
        self.ponding_area = np.array([n["ponding_area_m2"] for n in self.nodes])
        self.capacity = np.array([n["outflow_capacity_m3s"] for n in self.nodes])
        self.is_outfall = np.array([n["is_outfall"] for n in self.nodes])

        # Ground slope along the street, used for overland flow. We reuse the
        # slope of the outgoing pipe because sewers are laid to follow the
        # ground, so the two run downhill together.
        self.street_slope = np.array([max(n["outflow_slope"], 0.002) for n in self.nodes])

        # Where each node's water goes next. -1 means "leaves the system".
        self.downstream_idx = np.array([
            self.index_of[n["downstream"]] if n.get("downstream") else -1
            for n in self.nodes
        ])

        # Process order: headwaters first, outfall last. Doing it in this order
        # means that when we reach a node, everything draining into it has
        # already been worked out this timestep.
        self.topo_order = self._topological_order()

    def _topological_order(self) -> list:
        """
        Sort nodes so that every node comes after all of its upstream nodes.

        We do it by counting how far each node is from an outfall: a node's
        'depth' is one more than the deepest node draining into it. Sorting by
        that count descending gives us headwaters-first order.
        """
        distance_to_outfall = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            steps, current = 0, i
            while self.downstream_idx[current] >= 0 and steps < self.n:
                current = self.downstream_idx[current]
                steps += 1
            distance_to_outfall[i] = steps
        return list(np.argsort(-distance_to_outfall))

    # Commercial pipe sizes, in metres. A city cannot order a 0.63 m pipe -
    # it buys one of these. Any upgrade has to land on this ladder.
    STANDARD_DIAMETERS = [0.30, 0.375, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.50, 1.80]

    def capacity_with_intervention(self, node_ids: list, action: str = "upsize",
                                   steps: int = 1) -> tuple:
        """
        Return (modified capacity array, per-node description of what changed).

        There is more than one way to fix a drain, and which one works depends
        on WHY it is failing. Manning's equation shows both levers:

            Q = (1/n) · A · R^(2/3) · S^(1/2)

        A (size) and S (slope) are both in there. A pipe already at the largest
        commercial diameter cannot be fixed by buying a bigger one - the only
        lever left is slope, or a second pipe. Being able to say that is far
        more useful to a city than a generic "upgrade this".

            upsize  - climb the commercial diameter ladder
            regrade - re-lay the pipe at a steeper gradient
            relief  - lay a second pipe alongside (doubles capacity)
        """
        from backend.network import _manning_capacity

        capacity = self.capacity.copy()
        changes = {}

        for node_id in node_ids:
            if node_id not in self.index_of:
                continue
            i = self.index_of[node_id]
            node = self.nodes[i]
            if node["is_outfall"]:
                continue

            current_d = node["outflow_diameter_m"]
            current_s = node["outflow_slope"]
            before = self.capacity[i]
            ladder = self.STANDARD_DIAMETERS

            if action == "upsize":
                position = min(range(len(ladder)), key=lambda k: abs(ladder[k] - current_d))
                if position >= len(ladder) - 1:
                    # Honest dead end - say so rather than silently doing nothing.
                    changes[node_id] = {
                        "action": "upsize", "possible": False,
                        "reason": f"Already at the largest commercial pipe "
                                  f"({current_d:.2f} m). A bigger pipe is not an option — "
                                  f"this junction needs regrading or a relief sewer.",
                        "capacity_before": round(float(before), 3),
                        "capacity_after": round(float(before), 3),
                    }
                    continue
                new_d = ladder[min(position + steps, len(ladder) - 1)]
                capacity[i] = _manning_capacity(new_d, current_s)
                changes[node_id] = {
                    "action": "upsize", "possible": True,
                    "reason": f"Replace the {current_d:.2f} m pipe with {new_d:.2f} m.",
                    "capacity_before": round(float(before), 3),
                    "capacity_after": round(float(capacity[i]), 3),
                }

            elif action == "regrade":
                # Re-lay at a healthier gradient. 0.5% is a comfortable
                # self-cleansing grade; we never claim more than triple.
                new_s = min(max(current_s * 3.0, 0.005), 0.02)
                capacity[i] = _manning_capacity(current_d, new_s)
                changes[node_id] = {
                    "action": "regrade", "possible": True,
                    "reason": f"Re-lay the pipe at {new_s * 100:.2f}% instead of "
                              f"{current_s * 100:.2f}%. Steeper pipes also silt up less.",
                    "capacity_before": round(float(before), 3),
                    "capacity_after": round(float(capacity[i]), 3),
                }

            elif action == "relief":
                capacity[i] = before * 2.0
                changes[node_id] = {
                    "action": "relief", "possible": True,
                    "reason": f"Lay a second {current_d:.2f} m pipe alongside the existing one.",
                    "capacity_before": round(float(before), 3),
                    "capacity_after": round(float(capacity[i]), 3),
                }

        return capacity, changes

    def simulate(self, storm: Storm, collect_timeseries: bool = False,
                 capacity_override: np.ndarray = None) -> dict:
        """
        Run one storm through the network.

        Returns per-node results, plus (optionally) the full time series so the
        dashboard can animate the flood building and receding.

        `capacity_override` lets a caller swap in a modified set of pipe
        capacities - used by the what-if endpoint to test an upgrade without
        rebuilding the whole model.
        """
        total_minutes = storm.duration_min + RECESSION_MIN
        total_steps = int(math.ceil(total_minutes / DT_MIN))
        rainfall = rainfall_hyetograph(storm, total_steps)

        # -------------------------------------------------------------
        # Effective runoff coefficient.
        # If the ground is already saturated from previous days of rain, even
        # the soil and grass start behaving like tarmac. This is why the third
        # day of a wet spell floods worse than the first, and it is the single
        # most under-appreciated driver of urban flooding.
        # -------------------------------------------------------------
        c_effective = np.minimum(
            0.96,
            self.runoff_coeff + (1.0 - self.runoff_coeff) * storm.antecedent_saturation * 0.65,
        )

        # -------------------------------------------------------------
        # Effective pipe capacity.
        # Silt, sand and solid waste steal capacity from every pipe. A drain
        # half full of rubbish carries far less than its design flow - which
        # is why so much urban flooding is a maintenance problem, not a
        # capacity problem. Our sandbox lets you dial this in and watch.
        # -------------------------------------------------------------
        base_capacity = self.capacity if capacity_override is None else capacity_override
        effective_capacity = base_capacity * (1.0 - storm.blockage_factor)
        effective_capacity = np.where(self.is_outfall, 1e6, effective_capacity)
        effective_capacity = np.maximum(effective_capacity, 1e-4)

        # Running state, one value per node.
        ponded_volume = np.zeros(self.n)     # m3 sitting on the street
        upstream_inflow = np.zeros(self.n)   # m3/s arriving from upstream

        peak_depth = np.zeros(self.n)
        peak_surcharge = np.zeros(self.n)
        peak_inflow = np.zeros(self.n)
        flood_volume = np.zeros(self.n)
        minutes_flooded = np.zeros(self.n)
        time_to_flood = np.full(self.n, -1.0)

        timeseries = [] if collect_timeseries else None

        for step in range(total_steps):
            intensity = rainfall[step]

            # Rain landing on each sub-catchment, converted to a flow rate.
            # This is the Rational Method, applied to every node at once.
            local_runoff = c_effective * intensity * self.area_ha / 360.0

            next_upstream_inflow = np.zeros(self.n)

            for i in self.topo_order:
                # Everything trying to get through this junction right now:
                # fresh rain + water arriving from upstream + water already
                # ponded here that is trying to drain away.
                ponded_release = ponded_volume[i] / DT_SEC
                total_inflow = local_runoff[i] + upstream_inflow[i] + ponded_release

                cap = effective_capacity[i]

                # --- MINOR SYSTEM: the pipe takes what it can ---------------
                pipe_flow = min(total_inflow, cap)
                retained = max(0.0, (total_inflow - pipe_flow) * DT_SEC)

                # --- MAJOR SYSTEM: the rest runs down the street ------------
                # Once the water is deeper than the kerb it starts flowing
                # overland towards the next low point. Manning's equation for
                # a wide shallow channel gives us that flow rate.
                depth = retained / self.ponding_area[i]
                overland_flow = 0.0
                if depth > KERB_DEPTH_M:
                    head = depth - KERB_DEPTH_M
                    conveying_head = min(head, OVERLAND_MAX_HEAD_M)
                    overland_flow = (
                        (1.0 / STREET_ROUGHNESS)
                        * STREET_WIDTH_M
                        * (conveying_head ** (5.0 / 3.0))
                        * math.sqrt(self.street_slope[i])
                    )
                    # Cannot move more water than is actually standing there.
                    overland_flow = min(overland_flow, head * self.ponding_area[i] / DT_SEC)
                    retained -= overland_flow * DT_SEC

                ponded_volume[i] = max(0.0, retained)
                depth = min(MAX_DEPTH_M, ponded_volume[i] / self.ponding_area[i])

                # Record the worst values seen so far at this node.
                arriving = local_runoff[i] + upstream_inflow[i]
                surcharge = arriving / cap

                if depth > peak_depth[i]:
                    peak_depth[i] = depth
                if surcharge > peak_surcharge[i]:
                    peak_surcharge[i] = surcharge
                if arriving > peak_inflow[i]:
                    peak_inflow[i] = arriving

                if depth >= DEPTH_NUISANCE:
                    minutes_flooded[i] += DT_MIN
                    flood_volume[i] += max(0.0, total_inflow - pipe_flow) * DT_SEC
                    if time_to_flood[i] < 0:
                        time_to_flood[i] = step * DT_MIN

                # Hand both the piped flow and the street flow to the
                # downstream junction. They arrive at the start of the next
                # timestep - that one-step delay is our travel-time lag.
                # At an outfall the water simply leaves the system (the river).
                target = self.downstream_idx[i]
                if target >= 0:
                    next_upstream_inflow[target] += pipe_flow + overland_flow

            upstream_inflow = next_upstream_inflow

            if collect_timeseries:
                timeseries.append({
                    "minute": round(step * DT_MIN, 1),
                    "intensity_mmhr": round(float(intensity), 2),
                    "depths": [round(float(ponded_volume[i] / self.ponding_area[i]), 4)
                               for i in range(self.n)],
                    "flooded_count": int(np.sum(
                        (ponded_volume / self.ponding_area) >= DEPTH_NUISANCE)),
                })

        results = {}
        for i, node_id in enumerate(self.node_ids):
            results[node_id] = NodeResult(
                node_id=node_id,
                peak_depth_m=round(float(peak_depth[i]), 4),
                peak_surcharge_ratio=round(float(peak_surcharge[i]), 4),
                flood_volume_m3=round(float(flood_volume[i]), 2),
                minutes_flooded=float(minutes_flooded[i]),
                time_to_flood_min=float(time_to_flood[i]),
                peak_inflow_m3s=round(float(peak_inflow[i]), 4),
            )

        summary = {
            "nodes_flooded": int(np.sum(peak_depth >= DEPTH_NUISANCE)),
            "nodes_impassable": int(np.sum(peak_depth >= DEPTH_VEHICLE)),
            "total_flood_volume_m3": round(float(np.sum(flood_volume)), 1),
            "max_depth_m": round(float(np.max(peak_depth)), 3),
            "total_rainfall_mm": round(float(np.sum(rainfall) * DT_MIN / 60.0), 1),
            "simulated_minutes": total_steps * DT_MIN,
        }

        return {
            "storm": storm.as_dict(),
            "summary": summary,
            "nodes": results,
            "timeseries": timeseries,
        }


if __name__ == "__main__":
    # A quick self-test: run three storms of increasing severity and check the
    # flooding gets worse. If it does not, something is wrong with the physics.
    from backend.network import load_network

    model = HydraulicModel(load_network())

    scenarios = [
        ("Light shower  ", Storm(20, 45, 0.15, 0.05)),
        ("Heavy rain    ", Storm(60, 60, 0.35, 0.15)),
        ("Extreme storm ", Storm(110, 90, 0.70, 0.35)),
    ]

    print(f"{'Scenario':<16}{'Rain(mm)':>10}{'Flooded':>10}{'Impass.':>10}{'MaxDepth':>10}")
    print("-" * 56)
    for label, storm in scenarios:
        out = model.simulate(storm)
        s = out["summary"]
        print(f"{label:<16}{s['total_rainfall_mm']:>10.1f}{s['nodes_flooded']:>10}"
              f"{s['nodes_impassable']:>10}{s['max_depth_m']:>10.2f}")
