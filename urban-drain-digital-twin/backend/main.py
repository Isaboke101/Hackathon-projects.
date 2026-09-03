"""
main.py
=======
The FastAPI server. This is what the dashboard talks to.

Run it with:
    uvicorn backend.main:app --reload

Then open http://127.0.0.1:8000 in a browser.

Everything expensive - loading the network, the road graph and the trained
models - happens once at startup and is then reused. A request only has to do
the cheap part, which is why the sliders feel instant.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.hydraulics import (
    HydraulicModel, Storm, DEPTH_NUISANCE, DEPTH_PEDESTRIAN, DEPTH_VEHICLE,
)
from backend.network import load_network
from backend.predictor import FloodPredictor
from backend.predictor import analyse_interventions
from backend.routing import RoadNetwork, TRAVEL_MODES

app = FastAPI(
    title="Urban Drain Digital Twin",
    description="A virtual sandbox for urban stormwater. Simulate a storm, "
                "predict where it floods, and route around the water.",
    version="1.0.0",
)

# Allow the dashboard to call the API from anywhere. Fine for a hackathon
# demo; a production deployment would name the exact allowed origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load everything once, at import time.
# ---------------------------------------------------------------------------
print("Loading drainage network...")
NETWORK = load_network()

print("Building road network...")
ROADS = RoadNetwork(NETWORK)

print("Starting hydraulic model...")
HYDRAULICS = HydraulicModel(NETWORK)

print("Loading trained models...")
PREDICTOR = FloodPredictor(NETWORK)
if not PREDICTOR.available:
    print("  !! No trained model found. Run:  python -m backend.train")
else:
    print("  models loaded")

# Bottleneck ranking is the same every time. train.py precomputes it; if that
# file is missing we fall back to computing it on first request.
_BOTTLENECK_PATH = Path("models/bottlenecks.json")
_BOTTLENECK_CACHE = None


# ---------------------------------------------------------------------------
# Request shapes. Pydantic validates these for us and generates the API docs
# at /docs, which is a nice thing to show a judge.
# ---------------------------------------------------------------------------
class StormRequest(BaseModel):
    peak_intensity_mmhr: float = Field(60.0, ge=0, le=250, description="Peak rainfall rate")
    duration_min: float = Field(60.0, ge=5, le=360, description="How long it rains")
    antecedent_saturation: float = Field(0.3, ge=0, le=1, description="0 = dry ground, 1 = soaked")
    blockage_factor: float = Field(0.15, ge=0, le=0.9, description="Share of pipe capacity lost to silt and waste")

    def to_storm(self) -> Storm:
        return Storm(
            peak_intensity_mmhr=self.peak_intensity_mmhr,
            duration_min=self.duration_min,
            antecedent_saturation=self.antecedent_saturation,
            blockage_factor=self.blockage_factor,
        )


class RouteRequest(StormRequest):
    origin: str = Field(..., description="Origin road node id, e.g. R3_9")
    destination: str = Field(..., description="Destination road node id, e.g. R9_4")
    use_model: bool = Field(True, description="True = ML prediction, False = full physics simulation")
    mode: str = Field("car", description="Traveller: car, boda, foot or emergency")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Quick check that the server is alive and knows what it is holding."""
    return {
        "status": "ok",
        "nodes": len(NETWORK["nodes"]),
        "conduits": len(NETWORK["conduits"]),
        "road_intersections": ROADS.graph.number_of_nodes(),
        "road_segments": ROADS.graph.number_of_edges(),
        "model_loaded": PREDICTOR.available,
    }


@app.get("/api/network")
def get_network():
    """The drainage network itself, so the dashboard can draw the pipes."""
    return NETWORK


@app.get("/api/places")
def get_places():
    """Named origins and destinations for the routing panel."""
    return {"places": ROADS.points_of_interest}


@app.get("/api/metrics")
def get_metrics():
    """How well the models scored, straight from training."""
    if not PREDICTOR.metrics:
        raise HTTPException(404, "No metrics found. Run: python -m backend.train")
    return PREDICTOR.metrics


@app.get("/api/thresholds")
def get_thresholds():
    """The depth thresholds the whole system uses, in one place."""
    return {
        "nuisance_m": DEPTH_NUISANCE,
        "pedestrian_m": DEPTH_PEDESTRIAN,
        "vehicle_m": DEPTH_VEHICLE,
    }


@app.post("/api/simulate")
def simulate(request: StormRequest, timeseries: bool = False):
    """
    Run the full physics simulation. This is the ground truth - slower than
    the model, but it is what the model was trained to imitate.
    """
    result = HYDRAULICS.simulate(request.to_storm(), collect_timeseries=timeseries)
    return {
        "storm": result["storm"],
        "summary": result["summary"],
        "nodes": {node_id: vars(node) for node_id, node in result["nodes"].items()},
        "timeseries": result["timeseries"],
    }


@app.post("/api/predict")
def predict(request: StormRequest):
    """
    Score the network with the trained model. This is the fast path the
    dashboard uses while you are dragging a slider.
    """
    if not PREDICTOR.available:
        raise HTTPException(503, "No trained model. Run: python -m backend.train")
    return PREDICTOR.predict(request.to_storm())


@app.get("/api/bottlenecks")
def bottlenecks(top_n: int = 10):
    """
    The junctions that fail across the widest range of storms - i.e. the ones
    a city should fix first. Cached, because the answer never changes.
    """
    global _BOTTLENECK_CACHE
    if not PREDICTOR.available:
        raise HTTPException(503, "No trained model. Run: python -m backend.train")
    if _BOTTLENECK_CACHE is None:
        if _BOTTLENECK_PATH.exists():
            _BOTTLENECK_CACHE = json.loads(_BOTTLENECK_PATH.read_text())
        else:
            _BOTTLENECK_CACHE = PREDICTOR.rank_bottlenecks(top_n=20)
    return {"bottlenecks": _BOTTLENECK_CACHE[:top_n]}


@app.post("/api/route")
def route(request: RouteRequest):
    """
    The headline feature: given a storm, work out where the water will be and
    then find a way through it.

    Returns the route a normal satnav would give AND the route that avoids the
    water, so the difference is visible rather than asserted.
    """
    storm = request.to_storm()

    # Where will the water be? Either ask the model (fast) or run the full
    # simulation (slower, but ground truth). Being able to flip between them
    # live is a good way to show the model is faithful.
    if request.use_model and PREDICTOR.available:
        scored = PREDICTOR.predict(storm)
        depths = {nid: r["predicted_depth_m"] for nid, r in scored["nodes"].items()}
        source = "ml_model"
    else:
        simulated = HYDRAULICS.simulate(storm)
        depths = {nid: r.peak_depth_m for nid, r in simulated["nodes"].items()}
        source = "physics_simulation"

    ROADS.apply_flood(depths, mode=request.mode)

    try:
        routes = ROADS.route(request.origin, request.destination)
    except ValueError as error:
        raise HTTPException(400, str(error))

    fastest, safest = routes["fastest"], routes["safest"]

    return {
        "storm": storm.as_dict(),
        "depth_source": source,
        "fastest": fastest,
        "safest": safest,
        "identical": routes["identical"],
        "hazards": ROADS.hazard_segments(),
        "mode": request.mode,
        "mode_profile": TRAVEL_MODES.get(request.mode, TRAVEL_MODES["car"]),
        # The one-line summary the dashboard puts in front of the user.
        "benefit": {
            "extra_distance_m": safest["distance_m"] - fastest["distance_m"],
            "exposure_avoided_m": fastest["exposed_m"] - safest["exposed_m"],
            "impassable_avoided_m": fastest["impassable_m"] - safest["impassable_m"],
            "time_saved_min": round(fastest["estimated_minutes"] - safest["estimated_minutes"], 1),
        },
    }


@app.get("/api/modes")
def modes():
    """The kinds of traveller the router understands."""
    return {"modes": [{"key": key, **profile} for key, profile in TRAVEL_MODES.items()]}


@app.post("/api/compare-modes")
def compare_modes(request: RouteRequest):
    """
    Route the same journey for a car, a boda boda, a pedestrian and an
    emergency vehicle, all at once.

    During a bad storm the useful answer is rarely "here is the route" - it is
    "a car cannot get there, send a boda". This endpoint answers that directly.
    """
    storm = request.to_storm()

    if request.use_model and PREDICTOR.available:
        scored = PREDICTOR.predict(storm)
        depths = {nid: r["predicted_depth_m"] for nid, r in scored["nodes"].items()}
    else:
        simulated = HYDRAULICS.simulate(storm)
        depths = {nid: r.peak_depth_m for nid, r in simulated["nodes"].items()}

    try:
        comparison = ROADS.compare_modes(request.origin, request.destination, depths)
    except ValueError as error:
        raise HTTPException(400, str(error))

    return {"storm": storm.as_dict(), "modes": comparison}


# train.py writes this out, so normally we just read it from disk. Computing it
# live takes about two seconds of blocked server, which is unacceptable on the
# first page load of a demo.
_INTERVENTION_PATH = Path("models/interventions.json")
_INTERVENTION_CACHE = None
if _INTERVENTION_PATH.exists():
    _INTERVENTION_CACHE = json.loads(_INTERVENTION_PATH.read_text())
    print("Loaded precomputed repair plan")


@app.get("/api/interventions")
def interventions(top_n: int = 5):
    """
    A costed repair plan: which single fix removes the most flooding per
    shilling spent.

    Precomputed by train.py and read from disk. If that file is missing we
    compute it on demand, which is slower but never wrong.
    """
    global _INTERVENTION_CACHE
    if not PREDICTOR.available:
        raise HTTPException(503, "No trained model. Run: python -m backend.train")

    if _INTERVENTION_CACHE is None:
        candidates = [b["node_id"] for b in PREDICTOR.rank_bottlenecks(top_n=6)]
        _INTERVENTION_CACHE = analyse_interventions(NETWORK, HYDRAULICS, candidates)

    result = dict(_INTERVENTION_CACHE)
    result["best_value"] = _INTERVENTION_CACHE["best_value"][:top_n]
    return result


# ---------------------------------------------------------------------------
# Serve the dashboard itself.
# ---------------------------------------------------------------------------
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def dashboard():
        return FileResponse(FRONTEND_DIR / "index.html")
