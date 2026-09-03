"""
dataset.py
==========
Turns the physics engine into a machine-learning training set.

This is the "Synthetic Data Simulation" capability, and it is the answer to the
question every judge will ask: *where did you get your data?*

We do not have years of sensor readings from Nairobi's drains - almost nobody
does, anywhere. So we manufacture the data. We invent thousands of storms,
push each one through the hydraulic model, and record what happened at every
junction. Each (storm, junction) pair becomes one labelled training row.

Two things make this legitimate rather than circular:

  1. The labels come from PHYSICS, not from guesswork. The Rational Method and
     Manning's equation are the same tools a consulting engineer would use.

  2. We split train and test BY STORM, never by row. If the same storm had
     rows in both halves the model could memorise it and we would fool
     ourselves with a fake 99% score. This is the most common mistake in
     applied ML and we avoid it deliberately.
"""

import numpy as np
import pandas as pd

from backend.hydraulics import HydraulicModel, Storm, DEPTH_NUISANCE

# The static properties of a junction that never change from storm to storm.
STATIC_FEATURES = [
    "area_ha",
    "imperviousness",
    "runoff_coeff",
    "contributing_area_ha",
    "outflow_capacity_m3s",
    "outflow_diameter_m",
    "outflow_slope",
    "capacity_per_ha",
    "elevation_m",
    "ponding_area_m2",
    "direct_upstream_count",
]

# Properties of the storm itself.
STORM_FEATURES = [
    "peak_intensity_mmhr",
    "duration_min",
    "total_rainfall_mm",
    "antecedent_saturation",
    "blockage_factor",
]

# One engineered feature that combines the two. See the note in build_features.
ENGINEERED_FEATURES = [
    "demand_capacity_ratio",
]

FEATURE_COLUMNS = STATIC_FEATURES + STORM_FEATURES + ENGINEERED_FEATURES


def sample_storm(rng: np.random.Generator) -> Storm:
    """
    Draw one random storm.

    The ranges are chosen to straddle the point where the network starts to
    fail. Sampling only gentle storms would teach the model that nothing ever
    floods; sampling only extreme ones would teach it that everything does.
    We want it to learn exactly where the tipping point is.
    """
    return Storm(
        # 15 mm/hr is a light shower, 160 mm/hr is a once-in-a-generation
        # cloudburst. Nairobi's long rains routinely produce 50-90 mm/hr.
        peak_intensity_mmhr=float(rng.uniform(15, 160)),
        duration_min=float(rng.choice([20, 30, 45, 60, 90, 120, 180])),
        antecedent_saturation=float(rng.uniform(0.0, 0.95)),
        blockage_factor=float(rng.uniform(0.0, 0.55)),
    )


def build_features(nodes: list, storm: Storm, total_rainfall_mm: float) -> pd.DataFrame:
    """
    Assemble the feature table for one storm across every junction.

    This function is used in two places - here when building training data,
    and later at prediction time in predictor.py. Keeping it in one place is
    what stops the two drifting apart, which is a classic source of models
    that score well in training and fail in production.
    """
    rows = []
    for node in nodes:
        # ---------------------------------------------------------------
        # The one engineered feature: demand divided by capacity.
        #
        # DEMAND is the Rational Method estimate of peak flow arriving here:
        #     Q = C_effective * i * A_contributing / 360
        # CAPACITY is what the outgoing pipe can carry once we subtract the
        # share lost to silt and rubbish.
        #
        # A ratio above 1 means more water arrives than can leave. We hand
        # the model this ratio rather than making it rediscover the division,
        # because a drainage engineer would look at exactly this number first.
        # ---------------------------------------------------------------
        c_eff = min(0.96, node["runoff_coeff"]
                    + (1 - node["runoff_coeff"]) * storm.antecedent_saturation * 0.65)
        demand = c_eff * storm.peak_intensity_mmhr * node["contributing_area_ha"] / 360.0
        capacity = max(node["outflow_capacity_m3s"] * (1 - storm.blockage_factor), 1e-4)

        row = {name: node[name] for name in STATIC_FEATURES}
        row.update({
            "node_id": node["id"],
            "peak_intensity_mmhr": storm.peak_intensity_mmhr,
            "duration_min": storm.duration_min,
            "total_rainfall_mm": total_rainfall_mm,
            "antecedent_saturation": storm.antecedent_saturation,
            "blockage_factor": storm.blockage_factor,
            "demand_capacity_ratio": demand / capacity,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def generate_dataset(network: dict, n_scenarios: int = 2500, seed: int = 7) -> pd.DataFrame:
    """
    Run n_scenarios random storms and collect one labelled row per junction
    per storm. With 81 junctions and 2500 storms that is ~200,000 rows.
    """
    rng = np.random.default_rng(seed)
    model = HydraulicModel(network)
    nodes = network["nodes"]

    frames = []
    for scenario_id in range(n_scenarios):
        storm = sample_storm(rng)
        result = model.simulate(storm)

        frame = build_features(nodes, storm, result["summary"]["total_rainfall_mm"])
        frame["scenario_id"] = scenario_id

        # The labels, straight from the physics engine.
        frame["peak_depth_m"] = [result["nodes"][nid].peak_depth_m for nid in frame["node_id"]]
        frame["minutes_flooded"] = [result["nodes"][nid].minutes_flooded for nid in frame["node_id"]]
        frame["will_flood"] = (frame["peak_depth_m"] >= DEPTH_NUISANCE).astype(int)

        frames.append(frame)

        if (scenario_id + 1) % 250 == 0:
            print(f"  simulated {scenario_id + 1}/{n_scenarios} storms")

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    from pathlib import Path
    from backend.network import load_network

    print("Generating synthetic training data...")
    net = load_network()
    data = generate_dataset(net, n_scenarios=2500)

    Path("data").mkdir(exist_ok=True)
    data.to_csv("data/training_data.csv", index=False)

    flood_rate = data["will_flood"].mean()
    print(f"\nRows:            {len(data):,}")
    print(f"Storms:          {data['scenario_id'].nunique():,}")
    print(f"Flooded rows:    {data['will_flood'].sum():,} ({flood_rate:.1%})")
    print(f"Depth range:     {data['peak_depth_m'].min():.3f} - {data['peak_depth_m'].max():.2f} m")
    print("Saved to data/training_data.csv")
