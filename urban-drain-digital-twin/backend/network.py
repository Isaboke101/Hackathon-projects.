"""
network.py
==========
Builds the SYNTHETIC drainage network that the whole digital twin runs on.

Why synthetic?
--------------
Real municipal drainage as-built drawings are rarely digitised and almost never
public. Rather than let that block us, we generate a network whose *parameters*
(pipe diameters, slopes, catchment sizes, runoff coefficients) sit inside the
ranges used in standard urban drainage design. The geometry is invented; the
physics that runs on top of it is real.

The network is a "dendritic" tree: like a river system, many small pipes feed
into fewer big ones, and everything eventually reaches an outfall.

Everything is deterministic - we seed the random number generator - so the
network is identical every time you run it. That matters for a demo.
"""

import json
import math
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Study area: a box over central Nairobi.
# We picked Nairobi because it is our city and it floods badly every long-rains
# season. The coordinates are real; the pipes inside them are our own model.
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = -1.3080, -1.2660
LON_MIN, LON_MAX = 36.8020, 36.8520

# Grid resolution - 9 x 9 gives us 81 candidate junctions, a good size for a
# demo: big enough to look like a real network, small enough to reason about.
GRID_ROWS = 9
GRID_COLS = 9

SEED = 42

# Named zones, purely so the dashboard reads like a real city rather than a
# grid of numbers. Assigned by position within the study area.
ZONE_NAMES = [
    "Upper Hill", "Kilimani", "Community",
    "CBD West", "CBD Core", "CBD East",
    "Ngara", "Gikomba", "Industrial Area",
]


def _zone_for(row: int, col: int) -> str:
    """Map a grid cell onto one of nine named zones (3x3 blocks)."""
    band_r = min(2, row * 3 // GRID_ROWS)
    band_c = min(2, col * 3 // GRID_COLS)
    return ZONE_NAMES[band_r * 3 + band_c]


def _manning_capacity(diameter_m: float, slope: float, roughness: float = 0.013) -> float:
    """
    Manning's equation - the standard way engineers size a pipe.

        Q = (1 / n) * A * R^(2/3) * S^(1/2)

    where
        Q = flow the pipe can carry when running full   [m3/s]
        n = Manning roughness coefficient (0.013 for concrete)
        A = cross-sectional area of flow                [m2]
        R = hydraulic radius = A / wetted perimeter     [m]
        S = pipe slope (dimensionless, e.g. 0.004 = 0.4%)

    For a circular pipe running exactly full:
        A = pi * D^2 / 4
        R = D / 4
    """
    area = math.pi * (diameter_m ** 2) / 4.0
    hydraulic_radius = diameter_m / 4.0
    return (1.0 / roughness) * area * (hydraulic_radius ** (2.0 / 3.0)) * math.sqrt(slope)


def build_network(seed: int = SEED) -> dict:
    """
    Create the full drainage network and return it as a plain dictionary
    (which we then save as JSON so the frontend can draw it).
    """
    rng = random.Random(seed)

    # -------------------------------------------------------------------
    # STEP 1 - lay out junction nodes on a jittered grid.
    # Real manholes are not on a perfect grid, so we nudge each one a little.
    # -------------------------------------------------------------------
    nodes = {}
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            node_id = f"N{row}{col}"

            # Position: even spacing plus a small random offset.
            lat_frac = row / (GRID_ROWS - 1)
            lon_frac = col / (GRID_COLS - 1)
            jitter = 0.35 / GRID_ROWS
            lat = LAT_MIN + lat_frac * (LAT_MAX - LAT_MIN) + rng.uniform(-jitter, jitter) * (LAT_MAX - LAT_MIN)
            lon = LON_MIN + lon_frac * (LON_MAX - LON_MIN) + rng.uniform(-jitter, jitter) * (LON_MAX - LON_MIN)

            # ---------------------------------------------------------------
            # Ground elevation. Nairobi tilts downwards towards the north-east,
            # where the Nairobi River runs. We model that as a tilted plane,
            # then subtract two "bowls" - local depressions that collect water.
            # Those bowls are deliberate: every city has notorious dip points
            # and we want the model to discover them rather than be told.
            # ---------------------------------------------------------------
            base = 1720.0 - 55.0 * lat_frac - 35.0 * lon_frac

            # Depression 1 - a dip near the middle of the CBD.
            d1 = math.hypot(lat_frac - 0.50, lon_frac - 0.48)
            base -= 12.0 * math.exp(-(d1 ** 2) / 0.020)

            # Depression 2 - a low pocket towards the east.
            d2 = math.hypot(lat_frac - 0.72, lon_frac - 0.80)
            base -= 9.0 * math.exp(-(d2 ** 2) / 0.015)

            elevation = base + rng.uniform(-1.2, 1.2)

            # ---------------------------------------------------------------
            # Sub-catchment: the patch of ground that drains into this manhole.
            # area_ha        - hectares of land draining here
            # imperviousness - fraction covered in tarmac/roof/concrete (0-1).
            #                  The CBD is nearly all hard surface; the outskirts
            #                  keep some soil that can soak water up.
            # ---------------------------------------------------------------
            area_ha = round(rng.uniform(1.8, 6.5), 2)

            # Denser (more impervious) towards the CBD core in the middle.
            centrality = 1.0 - math.hypot(lat_frac - 0.5, lon_frac - 0.5) / 0.71
            imperviousness = round(min(0.95, max(0.35, 0.55 + 0.40 * centrality + rng.uniform(-0.08, 0.08))), 3)

            # Runoff coefficient C for the Rational Method. It is essentially
            # "what fraction of the rain that lands here becomes runoff".
            # Impervious ground gives roughly 0.90; open ground roughly 0.25.
            runoff_coeff = round(0.25 + 0.65 * imperviousness, 3)

            # The surface that floodwater actually spreads across at this
            # junction: roads, verges, car parks and open ground. In a dense
            # African city centre that is roughly 12-22% of the land area -
            # the rest is buildings, which water flows around rather than over.
            ponding_area_m2 = round(area_ha * 10000 * rng.uniform(0.12, 0.22), 1)

            nodes[node_id] = {
                "id": node_id,
                "row": row,
                "col": col,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "elevation_m": round(elevation, 2),
                "zone": _zone_for(row, col),
                "area_ha": area_ha,
                "imperviousness": imperviousness,
                "runoff_coeff": runoff_coeff,
                "ponding_area_m2": ponding_area_m2,
            }

    # -------------------------------------------------------------------
    # STEP 2 - connect each node to the lowest of its neighbours.
    # Water flows downhill, so this single rule builds a realistic drainage
    # tree with no cycles. Nodes that have no lower neighbour become outfalls.
    # -------------------------------------------------------------------
    conduits = []
    for node_id, node in nodes.items():
        row, col = node["row"], node["col"]

        # The northern edge of the study area is the Nairobi River corridor.
        # Anything reaching it discharges to the river, so those junctions are
        # outfalls. Having several outfalls instead of one matters: it splits
        # the city into separate drainage basins, which is how real cities
        # work, and it stops every drop of water funnelling to one corner.
        if row == GRID_ROWS - 1:
            node["is_outfall"] = True
            node["downstream"] = None
            continue

        # Look at the four cardinal neighbours.
        neighbours = []
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            n_row, n_col = row + d_row, col + d_col
            if 0 <= n_row < GRID_ROWS and 0 <= n_col < GRID_COLS:
                neighbours.append(nodes[f"N{n_row}{n_col}"])

        # Water flows downhill, so drain to the lowest neighbour.
        lower = [n for n in neighbours if n["elevation_m"] < node["elevation_m"] - 0.05]

        if lower:
            target = min(lower, key=lambda n: n["elevation_m"])
        else:
            # This node sits in a local depression - there is no downhill
            # neighbour. Real networks solve this by laying a pipe at the
            # minimum permitted grade towards the river. Those flat pipes are
            # notorious bottlenecks, and modelling them honestly is the point.
            downhill = [n for n in neighbours if n["row"] > row] or neighbours
            target = min(downhill, key=lambda n: n["elevation_m"])

        node["is_outfall"] = False
        node["downstream"] = target["id"]

        # Pipe length: great-circle distance between the two manholes.
        length_m = _haversine_m(node["lat"], node["lon"], target["lat"], target["lon"])

        # Pipe slope, floored at 0.2% - designers never lay a sewer flatter
        # than that because it would silt up.
        drop = node["elevation_m"] - target["elevation_m"]
        slope = max(0.002, drop / max(length_m, 1.0))

        conduits.append({
            "id": f"C_{node_id}_{target['id']}",
            "from": node_id,
            "to": target["id"],
            "length_m": round(length_m, 1),
            "slope": round(slope, 5),
        })

    # -------------------------------------------------------------------
    # STEP 3 - size the pipes.
    # A pipe near the edge of the network only drains its own patch; a pipe
    # near an outfall drains everything above it. So we first work out how
    # much land drains through each pipe, then choose a diameter from the
    # standard commercial sizes that is big enough for a "normal" storm.
    #
    # This is important: we size pipes for a moderate design storm, exactly
    # as a real municipality would. That is *why* extreme rain overwhelms
    # them, and it is what makes the twin's predictions meaningful.
    # -------------------------------------------------------------------
    contributing = _contributing_areas(nodes)
    standard_diameters = [0.30, 0.375, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.50, 1.80]

    for conduit in conduits:
        upstream_area_ha = contributing[conduit["from"]]

        # Design storm the network was (notionally) built for: 45 mm/hr, which
        # is roughly a 2-year return period storm for Nairobi. Rational Method:
        #   Q [m3/s] = C * i [mm/hr] * A [ha] / 360
        design_flow = 0.72 * 45.0 * upstream_area_ha / 360.0

        # Walk up the standard sizes until one can carry the design flow.
        chosen = standard_diameters[-1]
        for diameter in standard_diameters:
            if _manning_capacity(diameter, conduit["slope"]) >= design_flow:
                chosen = diameter
                break

        # A few pipes are deliberately undersized. Real networks are extended
        # piecemeal over decades and always contain legacy bottlenecks - this
        # is what our AI is meant to find.
        if rng.random() < 0.18 and standard_diameters.index(chosen) > 0:
            chosen = standard_diameters[standard_diameters.index(chosen) - 1]

        conduit["diameter_m"] = chosen
        conduit["capacity_m3s"] = round(_manning_capacity(chosen, conduit["slope"]), 4)
        conduit["upstream_area_ha"] = round(upstream_area_ha, 2)

    # -------------------------------------------------------------------
    # STEP 4 - attach useful summary numbers to each node so both the ML
    # model and the dashboard can use them without recomputing.
    # -------------------------------------------------------------------
    outflow_by_node = {c["from"]: c for c in conduits}
    upstream_count = {node_id: 0 for node_id in nodes}
    for conduit in conduits:
        upstream_count[conduit["to"]] += 1

    for node_id, node in nodes.items():
        pipe = outflow_by_node.get(node_id)
        node["contributing_area_ha"] = round(contributing[node_id], 2)
        node["outflow_capacity_m3s"] = pipe["capacity_m3s"] if pipe else 999.0
        node["outflow_diameter_m"] = pipe["diameter_m"] if pipe else 0.0
        node["outflow_slope"] = pipe["slope"] if pipe else 0.0
        node["direct_upstream_count"] = upstream_count[node_id]

        # "Capacity ratio" - how much pipe capacity exists per hectare that
        # drains through it. A low number is an early warning of a bottleneck.
        node["capacity_per_ha"] = round(
            node["outflow_capacity_m3s"] / max(node["contributing_area_ha"], 0.1), 4
        )

    return {
        "meta": {
            "city": "Nairobi, Kenya",
            "bbox": [LAT_MIN, LON_MIN, LAT_MAX, LON_MAX],
            "node_count": len(nodes),
            "conduit_count": len(conduits),
            "seed": seed,
            "note": "Synthetic network. Geometry is modelled; hydraulic parameters "
                    "follow standard urban drainage design ranges.",
        },
        "nodes": list(nodes.values()),
        "conduits": conduits,
    }


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two lat/lon points on the Earth's surface."""
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _contributing_areas(nodes: dict) -> dict:
    """
    For every node, total the catchment area of itself plus everything that
    drains into it. We do this by walking each node's path down to its outfall
    and adding its own area to each node along the way.
    """
    totals = {node_id: 0.0 for node_id in nodes}
    for node_id, node in nodes.items():
        current = node_id
        seen = set()
        while current is not None and current not in seen:
            seen.add(current)
            totals[current] += node["area_ha"]
            current = nodes[current].get("downstream")
    return totals


def save_network(path: str = "data/network.json", seed: int = SEED) -> dict:
    """Build the network and write it to disk as JSON."""
    network = build_network(seed)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(network, indent=2))
    return network


def load_network(path: str = "data/network.json") -> dict:
    """Read the network back from disk, building it first if it is missing."""
    file_path = Path(path)
    if not file_path.exists():
        return save_network(path)
    return json.loads(file_path.read_text())


if __name__ == "__main__":
    net = save_network()
    print(f"Network built: {net['meta']['node_count']} junctions, "
          f"{net['meta']['conduit_count']} pipes")
    outfalls = [n['id'] for n in net['nodes'] if n['is_outfall']]
    print(f"Outfalls: {outfalls}")
    caps = [c['capacity_m3s'] for c in net['conduits']]
    print(f"Pipe capacity range: {min(caps):.3f} - {max(caps):.3f} m3/s")
