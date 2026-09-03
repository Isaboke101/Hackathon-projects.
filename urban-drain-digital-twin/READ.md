# Urban Drain Digital Twin

A virtual sandbox that simulates urban stormwater in real time, predicts where
a city will flood before the rain falls, and routes people and emergency
vehicles around the water.

Built for the IEEE Tech Ignite Summer School hackathon.
Study area: central Nairobi, Kenya.

---

## Quick start

**Windows**

```
setup.bat      (once, ~4 minutes)
run.bat        (every time)
```

> **If setup fails with "Unknown compiler" or "Microsoft Visual C++ 14.0 is
> required":** your Python is newer than the scientific packages have prebuilt
> files for, so pip tried to compile them from C source. Install **Python
> 3.12** from [python.org](https://www.python.org/downloads/release/python-3128/),
> tick *Add Python to PATH*, and run `setup.bat` again — it finds 3.12
> automatically and rebuilds the environment. This is not a network problem
> and not a problem with the code.

**macOS / Linux**

```
./setup.sh
./run.sh
```

Then open **http://127.0.0.1:8000**

The interactive API documentation is at **http://127.0.0.1:8000/docs** — worth
showing a judge, it is generated automatically from the code.

---

## What it does

| Capability | How it works |
|---|---|
| **Synthetic data simulation** | Generates a drainage network and runs storms through it using the Rational Method and Manning's equation. 2,500 storms × 81 junctions = 202,500 labelled training rows. |
| **AI-driven prediction** | Gradient-boosted trees predict, for any storm, which junctions flood and how deep — before the rain starts. |
| **Intelligent routing** | Predicted flood depth is transferred onto a road graph; a weighted shortest-path search finds a way through — for a car, a boda boda, a pedestrian or a fire engine, each of which is stopped by different water. |
| **Costed repair plan** | Tries three different repairs at each bottleneck, measures what each removes, and ranks them by flooding removed per shilling. |
| **Interactive sandbox UI** | Leaflet dashboard: move a slider, the whole city re-scores in under 100 ms. |

### Two features worth demonstrating

**Who can actually get there.** "Impassable" is not one number. A pedestrian is
in danger at 0.15 m, a car floats at 0.30 m, a boda boda can pick a line along
the crown of the road, and a fire engine has real clearance. Click *Compare all
travellers* and the dashboard answers the question a control room actually
asks: not "what is the route" but "can anyone get there, and in what". In the
built-in **Water works cut off** scenario, no car can reach the water treatment
works — but a boda boda gets there in about seven minutes.

**Not every pipe can be fixed by making it bigger.** The two worst bottlenecks
in this network, N54 and N67, are already at the largest commercial pipe size
(1.80 m). Their problem is gradient, not diameter — they are laid at barely
0.2%. The tool says so explicitly and prices the alternatives instead. Laying a
relief sewer at N67 removes about a quarter of all city-wide flood volume for
roughly KES 27M. That is the single most fundable sentence in the project.

---

## How the physics works

Everything rests on two standard civil-engineering formulas. Learn these two
and you can defend the entire project.

**1. The Rational Method — how much rain becomes runoff**

```
Q = C · i · A / 360
```

- `Q` = peak runoff (m³/s)
- `C` = runoff coefficient — the fraction of rain that runs off instead of
  soaking in. Tarmac ≈ 0.9, grass ≈ 0.25.
- `i` = rainfall intensity (mm/hr)
- `A` = catchment area (hectares)

**2. Manning's equation — how much a pipe can carry**

```
Q = (1/n) · A · R^(2/3) · S^(1/2)
```

- `n` = roughness (0.013 for concrete pipe, 0.030 for a street)
- `A` = cross-sectional area of flow (m²)
- `R` = hydraulic radius = A / wetted perimeter; for a full circular pipe, `D/4`
- `S` = slope

**3. Put them together.** At every junction, every 5 minutes: water arrives
(rain + upstream flow), the pipe carries what it can, and whatever is left over
ponds on the street. That leftover water *is* the flood.

**4. Dual drainage.** Water that cannot fit in the pipe does not pile up
forever — it runs down the street to the next low point. Engineers call the
pipes the *minor system* and the streets the *major system*. We model both, and
we use Manning's equation again for the street, treating it as a wide shallow
channel. Ignoring the street half is the biggest mistake a simple flood model
can make.

---

## Project layout

```
urban-drain-digital-twin/
├── backend/
│   ├── network.py      Builds the synthetic drainage network
│   ├── hydraulics.py   The physics engine (Rational Method + Manning's)
│   ├── dataset.py      Runs thousands of storms to make training data
│   ├── train.py        Trains and honestly evaluates the models
│   ├── predictor.py    Loads models, predicts, ranks bottlenecks
│   ├── routing.py      Road graph and hazard-aware pathfinding
│   └── main.py         FastAPI server
├── frontend/
│   ├── index.html      Dashboard layout
│   ├── style.css       Dark control-room theme
│   ├── app.js          Map, charts, and all interaction
│   └── vendor/         Leaflet + Chart.js, bundled locally (see below)
├── data/               Generated network and training data
├── models/             Trained models, metrics, precomputed plans
└── demo/               Fallback demo recording — play this if the laptop dies
```

### API endpoints

| Endpoint | What it gives you |
|---|---|
| `GET /api/network` | The drainage network, for drawing |
| `GET /api/places` | Named origins and destinations |
| `GET /api/modes` | Traveller types and their depth limits |
| `GET /api/metrics` | Model scores from training |
| `GET /api/bottlenecks` | Chronically fragile junctions, ranked |
| `GET /api/interventions` | Costed repair plan, ranked by value |
| `POST /api/simulate` | Full physics simulation (ground truth) |
| `POST /api/predict` | ML prediction (the fast path) |
| `POST /api/route` | Safe vs direct route for one traveller |
| `POST /api/compare-modes` | The same journey for all four travellers |

All of them are browsable and testable at `/docs`.

### Rebuilding any stage on its own

```
python -m backend.network      # rebuild the drainage network
python -m backend.hydraulics   # self-test the physics
python -m backend.dataset      # regenerate training data (~35 s)
python -m backend.train        # retrain models + precompute the repair plan (~30 s)
python -m backend.predictor    # test predictions and bottleneck ranking
python -m backend.routing      # test the routing engine
```

`train.py` also writes `models/bottlenecks.json` and `models/interventions.json`.
Those are precomputed on purpose: computing them live blocks the server for
about two seconds on first page load, which is the worst possible moment.

Each file runs standalone and prints something useful. If a demo breaks, this
is how you find out which layer failed.

---

## Model results

| Metric | Value |
|---|---|
| Accuracy | 99.4% |
| F1 score | 0.985 |
| ROC-AUC | 0.9998 |
| Depth error (flooded junctions) | ±1.6 cm |
| Best baseline (tuned engineering rule) | F1 0.587 |
| Ensemble of 1,000 storms | 7.6× faster than simulating |

### Read this before you present

**Do not stand up and claim "99.4% accurate flood prediction for Nairobi."** It
is not true, and the first sharp judge will take the project apart with one
question.

What is actually true: the model is a **surrogate**. It was trained to
reproduce our own physics simulation, which is deterministic. Near-perfect
accuracy is the *expected and correct* result — it shows the surrogate
faithfully learned the simulator. It says nothing about real Nairobi, because
real-world accuracy depends on calibrating the simulator to the real network,
which needs survey data and rain-gauge records nobody has given us.

**Say this instead:**

> "The model reproduces our hydraulic simulation with 99.4% accuracy and 1.6 cm
> depth error. That is the correct result for a surrogate — it proves the AI
> layer works. The open question is calibrating the simulator to the real
> network, and that is exactly what we would do with a city's survey data."

That answer is stronger than the inflated claim, because it shows you
understand your own limitations. Judges reward that.

Two things in the results **are** genuinely impressive and you should lean on
them:

1. **F1 0.985 versus 0.587** for the best tuned engineering rule of thumb. We
   tuned the baseline in its own favour and still beat it comfortably. The
   model learned the cascade and timing effects a single ratio cannot capture.
2. **We split train and test by storm, not by row.** Every storm produces 81
   correlated rows; splitting rows at random would let the model memorise and
   score a fake 99%. Say this out loud — it signals you know what you are doing.

---

## Offline safety

**Leaflet and Chart.js are bundled in `frontend/vendor/`.** Nothing is loaded
from a CDN, so the dashboard works with no internet.

The one thing that needs internet is the OpenStreetMap background tiles. With
no connection, the tiles simply do not load and the dark background shows
through — **the network, the risk colours and the routes all still render**.
The demo degrades; it does not die.

Test this before you present: turn off your wifi, reload the page, and check
you are still happy with how it looks.

---

## Demo script (4 minutes)

**0:00 — Open on the map, no storm.**
"This is central Nairobi's drainage network — 81 junctions, 72 pipes, sized to
handle a normal storm. Green means the system is coping."

**0:30 — Drag the rainfall slider to ~85 mm/hr.**
"Now it's raining hard. Watch." *(Junctions turn orange and red.)*
"That took 40 milliseconds. The AI just re-scored every junction in the city."

**1:00 — Drag the blockage slider from 20% to 50%.**
"This is the slider that matters. It's not rainfall — it's how blocked the
drains are with silt and rubbish. Same storm, and flooding roughly doubles.
Most urban flooding is a maintenance problem, not a capacity problem, and this
tool lets a city prove that before the budget meeting."

**1:45 — Click "Long rains peak", then Calculate route.**
"Fire station to the central market during peak long rains. Grey dashed is what
a normal satnav gives you — straight through 1.2 km of road a car physically
cannot cross. Blue is ours: 679 metres longer, avoids all of it, and it's
actually two minutes *faster*, because you don't spend the time stuck in water."

**2:45 — Point at the bottleneck list.**
"This is the part a city would pay for. These junctions fail across the widest
range of storms — not in one bad storm, in most of them. And it tells you why:
N54's pipe is laid almost flat and drains 111 hectares. That's not a warning,
it's a work order."

**2:45b — Click "Water works cut off", then "Compare all travellers".**
"Here's the question a control room actually asks. Not 'what's the route' —
'can anyone get there'. No car can reach the water treatment works; 852 metres
of the only way in is too deep. A pedestrian definitely can't. But a boda boda
gets there in seven minutes, and so does a fire engine. In Nairobi that isn't
a hypothetical — the boda is already how most urgent things move."

**3:05 — Point at the costed repair plan.**
"And here's what a city buys. N67's pipe is already the biggest you can order —
1.8 metres — so you cannot fix it by making it bigger. Its problem is that it's
laid almost flat. A relief sewer alongside costs roughly 27 million shillings
and removes a quarter of all flooding in the model. That's a budget line, not a
research finding."

**3:25 — Untick "Use AI model".**
"Now we're running the full physics simulation instead of the model. Same
answer — 54 junctions, 2.95 metres versus 3.0. That's how we know the AI
learned the hydraulics rather than memorising noise."

**3:45 — Close.**
"Synthetic data, because sensor data doesn't exist. Physics, so the labels are
real. AI, so it's instant and explainable. And routing, so it saves someone
tonight."

---

## Questions you will be asked

**"Your data is fake. Why should I believe any of this?"**
The geometry is synthetic — we have no as-built drawings, and neither does
almost any African city. But the *physics* is not fake: the Rational Method and
Manning's equation are what a consulting engineer uses to size a real pipe, and
our parameters sit inside standard design ranges. Give us a city's network file
and the same code runs on real geometry tomorrow. The synthetic network is
scaffolding that let us build the whole system before the data exists.

**"Why use ML when you already have a physics model?"**
Three reasons, and we are honest that only two are strong today. First,
explainability — the model ranks which features drive risk, so we can tell a
city *why* a junction is fragile and therefore what to fix. Second, batch
speed — scoring 1,000 storms takes 1 second versus 8 seconds simulating, which
matters for ensemble forecasting. Third, and this is the real one: the moment a
city supplies sensor data, the same pipeline retrains on it and starts learning
the things our simplified physics misses.

**"How is this different from EPA SWMM?"** *(SWMM is the industry-standard free
hydraulic solver — know its name.)*
SWMM is far more accurate and we would not compete with it on hydraulics. SWMM
needs a fully surveyed network, takes minutes to hours per run, and gives you no
routing and no web dashboard. We are the layer above: instant what-if
exploration, risk explanation, and emergency routing. In a real deployment SWMM
would generate our training data instead of our simplified engine.

**"Isn't 99.4% accuracy suspicious?"**
Yes, and you are right to ask. See the section above — answer it with the
surrogate explanation, not a defence of the number.

**"What would you do with more time?"**
Calibrate against real rain-gauge records from the Kenya Meteorological
Department; ingest a live rainfall forecast so the prediction runs ahead of the
weather automatically; add an SMS alert layer, since that reaches far more
Nairobi residents than a web dashboard; and validate against historical flood
reports to check we are flagging the junctions that actually flooded.

**"Who pays for this?"**
Nairobi City County's water and sewerage company, and the road agencies. The
pitch is not "predict floods" — it is "spend your drain-cleaning budget where it
prevents the most damage", which is a maintenance-prioritisation product with a
measurable return.

---

## Known limitations

Say these before a judge finds them. It reads as competence, not weakness.

- The drainage network is synthetic. Real geometry would change specific
  results, though not the method.
- The hydraulic model is simplified: no backwater effects, no pressurised flow,
  no infiltration recovery between storms, and a one-timestep travel lag rather
  than a proper routing solution.
- Overland flow is a simplification of true 2-D surface flow.
- The road network is a modelled grid, not imported OpenStreetMap geometry —
  the obvious next upgrade, and `osmnx` would do it in about 20 lines.
- No calibration against observed floods, because we have no observations.
- Travel times assume a fixed base speed and no traffic.

---

## Built with

Python · FastAPI · scikit-learn · NetworkX · NumPy · pandas · Leaflet · Chart.js
