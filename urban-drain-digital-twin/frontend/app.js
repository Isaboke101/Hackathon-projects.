/* ==========================================================================
   Urban Drain Digital Twin - dashboard logic
   --------------------------------------------------------------------------
   Talks to the FastAPI backend, draws the drainage network on a Leaflet map,
   and re-colours everything whenever the storm changes.

   The whole file is plain JavaScript with no build step. Open the page and it
   runs - which matters when you have three nights.
   ========================================================================== */

const API = "";  // same origin as the page

// Colours must match the legend in index.html.
const BAND_COLOURS = {
  safe:     "#22c55e",
  low:      "#eab308",
  moderate: "#f97316",
  high:     "#ef4444",
  severe:   "#a21caf",
};

const PRESETS = {
  light:     { intensity: 25,  duration: 45,  saturation: 15, blockage: 10 },
  heavy:     { intensity: 70,  duration: 60,  saturation: 40, blockage: 20 },
  longrains: { intensity: 105, duration: 120, saturation: 70, blockage: 35 },
  extreme:   { intensity: 150, duration: 90,  saturation: 85, blockage: 50 },
  // The demo scenario: in this storm no car can reach the water works, but a
  // boda boda gets there in about seven minutes. Selecting it also sets the
  // route, so one click sets up the whole demonstration.
  cutoff: {
    intensity: 130, duration: 120, saturation: 75, blockage: 45,
    origin: "R9_4", destination: "R10_1", mode: "car",
  },
};

// ---------------------------------------------------------------- app state
const state = {
  network: null,
  places: [],
  thresholds: { nuisance_m: 0.10, pedestrian_m: 0.15, vehicle_m: 0.30 },
  prediction: null,
  nodeMarkers: {},
  pipeLines: [],
  hazardLayer: null,
  routeLayer: null,
  placeLayer: null,
  chart: null,
  map: null,
  pendingRequest: null,
  modes: [],
  mode: "car",
};

// ------------------------------------------------------------ tiny helpers
const $ = (id) => document.getElementById(id);

/** Read the four sliders into the shape the API expects. */
function currentStorm() {
  return {
    peak_intensity_mmhr: Number($("intensity").value),
    duration_min: Number($("duration").value),
    antecedent_saturation: Number($("saturation").value) / 100,
    blockage_factor: Number($("blockage").value) / 100,
  };
}

async function postJSON(path, body) {
  const response = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

async function getJSON(path) {
  const response = await fetch(API + path);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

/**
 * Wait until the user stops moving a slider before calling the API.
 * Without this, dragging a slider fires dozens of requests and the map
 * flickers as out-of-order responses land.
 */
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

// ==========================================================================
// MAP
// ==========================================================================
function initMap() {
  const [latMin, lonMin, latMax, lonMax] = state.network.meta.bbox;

  state.map = L.map("map", { zoomControl: true, attributionControl: true })
    .fitBounds([[latMin, lonMin], [latMax, lonMax]]);

  // Street tiles from OpenStreetMap. If there is no internet the tiles simply
  // will not load and the dark background shows through - the network, the
  // risk colours and the routes all still render. The demo degrades, it does
  // not die.
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    opacity: 0.42,
    attribution: '&copy; OpenStreetMap contributors',
  }).addTo(state.map);

  state.hazardLayer = L.layerGroup().addTo(state.map);
  state.routeLayer  = L.layerGroup().addTo(state.map);
  state.placeLayer  = L.layerGroup().addTo(state.map);

  drawPipes();
  drawNodes();
  drawPlaces();
}

/** Draw the drainage pipes, thicker where they carry more. */
function drawPipes() {
  const byId = Object.fromEntries(state.network.nodes.map((n) => [n.id, n]));

  state.network.conduits.forEach((conduit) => {
    const from = byId[conduit.from];
    const to = byId[conduit.to];

    const line = L.polyline(
      [[from.lat, from.lon], [to.lat, to.lon]],
      {
        color: "#3b82f6",
        weight: Math.max(1.2, Math.min(5, conduit.diameter_m * 3)),
        opacity: 0.42,
      }
    ).bindPopup(
      `<b>Pipe ${conduit.id}</b><br>` +
      `Diameter: ${conduit.diameter_m.toFixed(2)} m<br>` +
      `Length: ${conduit.length_m.toFixed(0)} m<br>` +
      `Slope: ${(conduit.slope * 100).toFixed(2)}%<br>` +
      `Capacity: ${conduit.capacity_m3s.toFixed(2)} m&sup3;/s<br>` +
      `Drains: ${conduit.upstream_area_ha.toFixed(1)} ha`
    );

    line.addTo(state.map);
    state.pipeLines.push(line);
  });
}

/** Draw a circle for every junction. Colour and size change with the storm. */
function drawNodes() {
  state.network.nodes.forEach((node) => {
    const marker = L.circleMarker([node.lat, node.lon], {
      radius: 5,
      color: "#0b1220",
      weight: 1,
      fillColor: BAND_COLOURS.safe,
      fillOpacity: 0.85,
    }).addTo(state.map);

    marker.on("click", () => showNodeDetail(node.id));
    state.nodeMarkers[node.id] = marker;
  });
}

/** Draw the named landmarks used as routing origins and destinations. */
function drawPlaces() {
  state.placeLayer.clearLayers();
  state.places.forEach((place) => {
    L.marker([place.lat, place.lon], {
      icon: L.divIcon({
        className: "",
        html: `<div style="background:#0b1220;border:2px solid #38bdf8;border-radius:50%;
                 width:11px;height:11px"></div>`,
        iconSize: [11, 11],
        iconAnchor: [5, 5],
      }),
    }).bindTooltip(place.label, { direction: "top", offset: [0, -6] })
      .addTo(state.placeLayer);
  });
}

// ==========================================================================
// SCORING - the main loop
// ==========================================================================
const refresh = debounce(async () => {
  $("loading").hidden = false;
  const started = performance.now();

  try {
    const storm = currentStorm();
    const useModel = $("useModel").checked;

    // The AI path scores the network directly. The physics path runs the full
    // simulation. Both return a depth per junction, so the map does not care
    // which produced it - which is exactly the point of the toggle.
    let nodes, summary;
    if (useModel) {
      const result = await postJSON("/api/predict", storm);
      nodes = result.nodes;
      summary = result.summary;
    } else {
      const result = await postJSON("/api/simulate", storm);
      nodes = {};
      Object.entries(result.nodes).forEach(([id, node]) => {
        nodes[id] = {
          node_id: id,
          predicted_depth_m: node.peak_depth_m,
          flood_probability: node.peak_depth_m >= state.thresholds.nuisance_m ? 1 : 0,
          risk_band: bandForDepth(node.peak_depth_m),
        };
      });
      summary = {
        at_risk: result.summary.nodes_flooded,
        impassable: result.summary.nodes_impassable,
        max_predicted_depth_m: result.summary.max_depth_m,
      };
    }

    state.prediction = nodes;
    paintNodes(nodes);
    updateStats(summary, performance.now() - started);
    updateChart(storm);
  } catch (error) {
    console.error(error);
    alert("Could not score the network:\n" + error.message);
  } finally {
    $("loading").hidden = true;
  }
}, 180);

/** Same banding rule the backend uses, for the physics path. */
function bandForDepth(depth) {
  if (depth >= state.thresholds.vehicle_m) return "severe";
  if (depth >= state.thresholds.pedestrian_m) return "high";
  if (depth >= state.thresholds.nuisance_m) return "moderate";
  if (depth > 0.02) return "low";
  return "safe";
}

/** Recolour and resize every junction marker. */
function paintNodes(nodes) {
  Object.entries(nodes).forEach(([id, result]) => {
    const marker = state.nodeMarkers[id];
    if (!marker) return;

    const depth = result.predicted_depth_m;
    marker.setStyle({
      fillColor: BAND_COLOURS[result.risk_band] || BAND_COLOURS.safe,
      fillOpacity: result.risk_band === "safe" ? 0.55 : 0.92,
    });
    // Deeper water draws a bigger dot, so severity reads at a glance.
    marker.setRadius(4 + Math.min(9, depth * 11));
  });
}

function updateStats(summary, elapsedMs) {
  $("statAtRisk").textContent = summary.at_risk;
  $("statImpassable").textContent = summary.impassable;
  $("statDepth").textContent = summary.max_predicted_depth_m.toFixed(2) + " m";
  $("statLatency").textContent = Math.round(elapsedMs) + " ms";

  $("statImpassable").style.color = summary.impassable > 0 ? BAND_COLOURS.severe : "";
  $("statAtRisk").style.color =
    summary.at_risk > 20 ? BAND_COLOURS.high :
    summary.at_risk > 5  ? BAND_COLOURS.moderate : "";
}

// ==========================================================================
// RAINFALL CHART
// ==========================================================================
/**
 * Redraw the storm profile.
 *
 * We rebuild the same single-peak design storm the backend uses so the user
 * can see the shape of what they configured, not just its peak number.
 */
function updateChart(storm) {
  const dt = 5;
  const rainSteps = Math.ceil(storm.duration_min / dt);
  const totalSteps = Math.ceil((storm.duration_min + 90) / dt);

  const labels = [];
  const values = [];
  for (let step = 0; step < totalSteps; step++) {
    labels.push(step * dt);
    if (step < rainSteps) {
      const t = (step + 0.5) / rainSteps;
      const peak = 0.4;
      let shape = t <= peak ? t / peak : (1 - t) / (1 - peak);
      shape = Math.max(0.10, Math.min(1, shape));
      values.push(+(shape * storm.peak_intensity_mmhr).toFixed(1));
    } else {
      values.push(0);
    }
  }

  // Total depth of rain delivered - the number people actually recognise.
  // "85 mm/hr" means little; "62 mm of rain in two hours" is a headline.
  const totalMm = values.reduce((sum, v) => sum + v, 0) * dt / 60;
  $("chartCaption").textContent =
    `${totalMm.toFixed(0)} mm of rain over ${storm.duration_min} minutes. ` +
    `Nairobi's wettest days deliver 60-100 mm.`;

  if (state.chart) {
    state.chart.data.labels = labels;
    state.chart.data.datasets[0].data = values;
    state.chart.update("none");
    return;
  }

  state.chart = new Chart($("stormChart"), {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Rainfall (mm/hr)",
        data: values,
        borderColor: "#38bdf8",
        backgroundColor: "rgba(56,189,248,.18)",
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          title: { display: true, text: "minutes", color: "#64748b", font: { size: 10 } },
          ticks: { color: "#64748b", maxTicksLimit: 7, font: { size: 10 } },
          grid: { color: "rgba(36,51,82,.5)" },
        },
        y: {
          title: { display: true, text: "mm/hr", color: "#64748b", font: { size: 10 } },
          ticks: { color: "#64748b", font: { size: 10 } },
          grid: { color: "rgba(36,51,82,.5)" },
        },
      },
    },
  });
}

// ==========================================================================
// ROUTING
// ==========================================================================
/** Build the traveller picker from whatever modes the backend offers. */
function renderModeButtons() {
  $("modeGrid").innerHTML = state.modes.map((mode) => `
    <button class="mode-btn ${mode.key === state.mode ? "active" : ""}"
            data-mode="${mode.key}" title="${mode.note.replace(/"/g, "&quot;")}">
      ${mode.label}
      <small>${mode.depth_limit_m.toFixed(2)} m limit</small>
    </button>`).join("");

  $("modeGrid").querySelectorAll(".mode-btn").forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      $("modeGrid").querySelectorAll(".mode-btn")
        .forEach((b) => b.classList.toggle("active", b === button));
      // Re-route immediately if a route is already on screen, so switching
      // traveller feels instant rather than requiring another click.
      if (!$("routeResult").hidden) calculateRoute();
    });
  });
}

async function calculateRoute() {
  const button = $("routeBtn");
  button.disabled = true;
  button.textContent = "Calculating…";

  try {
    const payload = {
      ...currentStorm(),
      origin: $("origin").value,
      destination: $("destination").value,
      use_model: $("useModel").checked,
      mode: state.mode,
    };
    const result = await postJSON("/api/route", payload);

    drawRoutes(result);
    renderRouteResult(result);
    $("modeCompare").hidden = true;
  } catch (error) {
    alert("Could not calculate a route:\n" + error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Calculate route";
  }
}

/**
 * Route the same journey for every kind of traveller.
 *
 * This is the question that actually matters in a storm: not "what is the
 * route" but "can anyone get there, and in what".
 */
async function compareTravellers() {
  const button = $("compareBtn");
  button.disabled = true;
  button.textContent = "Comparing…";

  try {
    const result = await postJSON("/api/compare-modes", {
      ...currentStorm(),
      origin: $("origin").value,
      destination: $("destination").value,
      use_model: $("useModel").checked,
      mode: state.mode,
    });

    const rows = Object.entries(result.modes).map(([key, entry]) => {
      const route = entry.safest;
      const blocked = !route.passable;
      return `
        <div class="mode-row ${blocked ? "blocked" : ""}">
          <div>
            <div class="mr-label">${entry.label}</div>
            <!-- Deliberately not showing max depth here: it is the depth of
                 the water, which is the same whoever is travelling. What
                 differs is whether that stops you, and the verdict says so. -->
            <div class="mr-sub">
              ${(route.distance_m / 1000).toFixed(2)} km ·
              ${route.estimated_minutes} min ·
              stopped by ${entry.depth_limit_m.toFixed(2)} m
            </div>
          </div>
          <div class="mr-verdict">
            <b>${blocked ? "blocked" : "gets through"}</b>
            ${blocked ? `<span>${route.impassable_m} m impassable</span>` : ""}
          </div>
        </div>`;
    }).join("");

    $("modeCompare").innerHTML = `<h4>Who can actually get there</h4>${rows}`;
    $("modeCompare").hidden = false;
  } catch (error) {
    alert("Could not compare travellers:\n" + error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Compare all travellers";
  }
}

function drawRoutes(result) {
  state.routeLayer.clearLayers();
  state.hazardLayer.clearLayers();

  // Flooded roads first, so the routes draw on top of them.
  if ($("showHazards").checked) {
    result.hazards.forEach((hazard) => {
      L.polyline(hazard.coordinates, {
        color: hazard.impassable ? "#ef4444" : "#f97316",
        weight: hazard.impassable ? 5 : 3,
        opacity: 0.75,
      }).bindTooltip(
        `${hazard.depth_m.toFixed(2)} m deep` +
        (hazard.impassable ? " &mdash; impassable" : ""),
        { sticky: true }
      ).addTo(state.hazardLayer);
    });
  }

  // The direct route: what a normal satnav would tell you.
  L.polyline(result.fastest.coordinates, {
    color: "#94a3c4",
    weight: 4,
    opacity: 0.85,
    dashArray: "7,7",
  }).bindTooltip("Direct route (ignores flooding)").addTo(state.routeLayer);

  // The safe route, drawn last so it sits on top.
  L.polyline(result.safest.coordinates, {
    color: "#38bdf8",
    weight: 5,
    opacity: 0.95,
  }).bindTooltip("Safe route").addTo(state.routeLayer);

  const bounds = L.latLngBounds([
    ...result.fastest.coordinates,
    ...result.safest.coordinates,
  ]);
  state.map.fitBounds(bounds, { padding: [55, 55] });
}

function renderRouteResult(result) {
  const box = $("routeResult");
  const { fastest, safest, benefit } = result;

  let headline;
  if (result.identical) {
    headline = `<div class="route-headline">
        The direct route is <strong>already the safest</strong> option in this storm.
      </div>`;
  } else if (benefit.impassable_avoided_m > 0) {
    headline = `<div class="route-headline">
        Detour avoids <strong>${benefit.impassable_avoided_m} m</strong> of road a
        vehicle cannot cross, for ${benefit.extra_distance_m} m extra distance.
      </div>`;
  } else {
    headline = `<div class="route-headline">
        Detour cuts flood exposure by <strong>${benefit.exposure_avoided_m} m</strong>
        for ${benefit.extra_distance_m} m extra distance.
      </div>`;
  }

  if (!safest.passable) {
    headline = `<div class="route-headline warn">
        <strong>No safe route exists.</strong> Every way through crosses
        ${safest.impassable_m} m of impassable water. Air or boat access only.
      </div>`;
  }

  box.innerHTML = headline + `
    <div class="route-compare">
      <div class="route-col">
        <h4>Direct</h4>
        <div class="big">${(fastest.distance_m / 1000).toFixed(2)} km</div>
        <div class="sub">
          ${fastest.estimated_minutes} min<br>
          ${fastest.exposed_m} m in water<br>
          max ${fastest.max_depth_m.toFixed(2)} m
        </div>
      </div>
      <div class="route-col safe">
        <h4>Safe route</h4>
        <div class="big">${(safest.distance_m / 1000).toFixed(2)} km</div>
        <div class="sub">
          ${safest.estimated_minutes} min<br>
          ${safest.exposed_m} m in water<br>
          max ${safest.max_depth_m.toFixed(2)} m
        </div>
      </div>
    </div>
    <div class="route-row" style="margin-top:9px">
      <span class="k">Depth source</span>
      <span class="v">${result.depth_source === "ml_model" ? "AI model" : "Physics sim"}</span>
    </div>`;

  box.hidden = false;
}

// ==========================================================================
// SIDE PANELS
// ==========================================================================
async function loadBottlenecks() {
  try {
    const { bottlenecks } = await getJSON("/api/bottlenecks?top_n=6");
    $("bottleneckList").innerHTML = bottlenecks.map((item) => `
      <li data-node="${item.node_id}">
        <div class="bn-head">
          <span class="bn-zone">${item.zone} &middot; ${item.node_id}</span>
          <span class="bn-score">${Math.round(item.vulnerability * 100)}%</span>
        </div>
        <div class="bn-why">
          Fails ${item.storms_failed} of ${item.storms_tested} design storms.
          ${item.diagnosis}
        </div>
      </li>`).join("");

    // Clicking a bottleneck flies the map to it.
    $("bottleneckList").querySelectorAll("li").forEach((li) => {
      li.addEventListener("click", () => {
        const node = state.network.nodes.find((n) => n.id === li.dataset.node);
        state.map.flyTo([node.lat, node.lon], 16, { duration: 0.7 });
        showNodeDetail(node.id);
      });
    });
  } catch (error) {
    $("bottleneckList").innerHTML =
      `<li style="padding-left:0">Train the model first:<br>
       <code>python -m backend.train</code></li>`;
  }
}

async function loadRepairPlan() {
  try {
    const plan = await getJSON("/api/interventions?top_n=4");

    $("repairList").innerHTML = plan.best_value.map((option) => `
      <li>
        <div class="rp-head">
          <span class="rp-where">${option.zone} &middot; ${option.node_id}</span>
          <span class="rp-cut">&minus;${option.flood_volume_cut_pct}% flooding</span>
        </div>
        <div class="rp-why">${option.reason}</div>
        <div class="rp-cost">
          <b>KES ${(option.indicative_cost_kes / 1e6).toFixed(1)}M</b> indicative &middot;
          ${option.m3_per_million_kes.toLocaleString()} m&sup3; removed per million
        </div>
      </li>`).join("");

    $("costCaveat").textContent = plan.cost_caveat;
  } catch (error) {
    $("repairList").innerHTML =
      `<li style="padding-left:0">Train the model first:<br>
       <code>python -m backend.train</code></li>`;
    $("costCaveat").textContent = "";
  }
}

async function loadMetrics() {
  try {
    const metrics = await getJSON("/api/metrics");
    const c = metrics.classifier;
    const r = metrics.regressor;
    const best = c.baselines.tuned_threshold;

    $("modelMetrics").innerHTML = `
      <div class="metric-row"><span class="k">Accuracy</span>
        <span class="v">${(c.accuracy * 100).toFixed(1)}%</span></div>
      <div class="metric-row"><span class="k">F1 score</span>
        <span class="v">${c.f1.toFixed(3)}</span></div>
      <div class="metric-row"><span class="k">ROC-AUC</span>
        <span class="v">${c.roc_auc.toFixed(3)}</span></div>
      <div class="metric-row"><span class="k">Depth error</span>
        <span class="v">&plusmn;${(r.mae_on_flooded_m * 100).toFixed(1)} cm</span></div>
      <div class="metric-row"><span class="k">Best rule baseline</span>
        <span class="v">F1 ${best.f1.toFixed(3)}</span></div>
      <div class="metric-row"><span class="k">Ensemble speed-up</span>
        <span class="v">${metrics.speed.ensemble_1000_storms.speedup}&times;</span></div>
      <div class="metric-note">${metrics.honest_caveat}</div>`;
  } catch (error) {
    $("modelMetrics").innerHTML =
      `<p class="hint">No metrics yet. Run <code>python -m backend.train</code>.</p>`;
  }
}

function showNodeDetail(nodeId) {
  const node = state.network.nodes.find((n) => n.id === nodeId);
  const result = state.prediction ? state.prediction[nodeId] : null;
  if (!node) return;

  const band = result ? result.risk_band : "safe";
  const colour = BAND_COLOURS[band];

  $("nodeDetailBody").innerHTML = `
    <h3>${node.zone} &middot; ${node.id}</h3>
    <div class="nd-sub">Drainage junction</div>
    <span class="band-pill" style="background:${colour}22;color:${colour};
      border:1px solid ${colour}66">${band}</span>
    ${result ? `
      <div class="route-row"><span class="k">Predicted depth</span>
        <span class="v">${result.predicted_depth_m.toFixed(2)} m</span></div>
      <div class="route-row"><span class="k">Flood probability</span>
        <span class="v">${Math.round(result.flood_probability * 100)}%</span></div>
    ` : ""}
    <div class="route-row"><span class="k">Catchment</span>
      <span class="v">${node.area_ha} ha</span></div>
    <div class="route-row"><span class="k">Drains in total</span>
      <span class="v">${node.contributing_area_ha} ha</span></div>
    <div class="route-row"><span class="k">Hard surface</span>
      <span class="v">${Math.round(node.imperviousness * 100)}%</span></div>
    <div class="route-row"><span class="k">Outgoing pipe</span>
      <span class="v">${node.outflow_diameter_m.toFixed(2)} m</span></div>
    <div class="route-row"><span class="k">Pipe capacity</span>
      <span class="v">${node.outflow_capacity_m3s.toFixed(2)} m&sup3;/s</span></div>
    <div class="route-row"><span class="k">Pipe slope</span>
      <span class="v">${(node.outflow_slope * 100).toFixed(2)}%</span></div>
    <div class="route-row"><span class="k">Ground level</span>
      <span class="v">${node.elevation_m.toFixed(1)} m</span></div>`;

  $("nodeDetail").hidden = false;
}

// ==========================================================================
// WIRING
// ==========================================================================
function bindControls() {
  // Sliders: update the number beside the label, then re-score.
  [["intensity", "intensityOut"], ["duration", "durationOut"],
   ["saturation", "saturationOut"], ["blockage", "blockageOut"]]
    .forEach(([input, output]) => {
      $(input).addEventListener("input", () => {
        $(output).textContent = $(input).value;
        clearPresetHighlight();
        refresh();
      });
    });

  $("useModel").addEventListener("change", refresh);

  document.querySelectorAll(".preset").forEach((button) => {
    button.addEventListener("click", () => {
      const preset = PRESETS[button.dataset.preset];
      $("intensity").value = preset.intensity;
      $("duration").value = preset.duration;
      $("saturation").value = preset.saturation;
      $("blockage").value = preset.blockage;

      $("intensityOut").textContent = preset.intensity;
      $("durationOut").textContent = preset.duration;
      $("saturationOut").textContent = preset.saturation;
      $("blockageOut").textContent = preset.blockage;

      // Some presets also set up the journey they are meant to demonstrate.
      if (preset.origin) $("origin").value = preset.origin;
      if (preset.destination) $("destination").value = preset.destination;
      if (preset.mode) {
        state.mode = preset.mode;
        renderModeButtons();
      }

      clearPresetHighlight();
      button.classList.add("active");
      refresh();
    });
  });

  $("routeBtn").addEventListener("click", calculateRoute);
  $("compareBtn").addEventListener("click", compareTravellers);
  $("closeDetail").addEventListener("click", () => { $("nodeDetail").hidden = true; });

  $("showPipes").addEventListener("change", (event) => {
    state.pipeLines.forEach((line) => {
      event.target.checked ? line.addTo(state.map) : state.map.removeLayer(line);
    });
  });
  $("showHazards").addEventListener("change", (event) => {
    event.target.checked
      ? state.hazardLayer.addTo(state.map)
      : state.map.removeLayer(state.hazardLayer);
  });
  $("showPlaces").addEventListener("change", (event) => {
    event.target.checked
      ? state.placeLayer.addTo(state.map)
      : state.map.removeLayer(state.placeLayer);
  });
}

function clearPresetHighlight() {
  document.querySelectorAll(".preset").forEach((b) => b.classList.remove("active"));
}

function fillPlaceDropdowns() {
  const options = state.places
    .map((place) => `<option value="${place.node_id}">${place.label}</option>`)
    .join("");
  $("origin").innerHTML = options;
  $("destination").innerHTML = options;

  // Default to a pair that crosses the flood-prone middle of the city, so the
  // first click of the demo shows a real difference between the two routes.
  $("origin").value = "R3_9";      // Fire Station
  $("destination").value = "R9_4"; // Central Market
}

// ------------------------------------------------------------------- start
(async function start() {
  try {
    const [network, places, thresholds, modes] = await Promise.all([
      getJSON("/api/network"),
      getJSON("/api/places"),
      getJSON("/api/thresholds"),
      getJSON("/api/modes"),
    ]);

    state.network = network;
    state.places = places.places;
    state.thresholds = thresholds;
    state.modes = modes.modes;

    initMap();
    fillPlaceDropdowns();
    renderModeButtons();
    bindControls();

    // The repair plan runs a lot of simulations, so let it arrive on its own
    // rather than holding up the first paint of the map.
    await Promise.all([loadBottlenecks(), loadMetrics()]);
    refresh();
    loadRepairPlan();
  } catch (error) {
    document.body.innerHTML =
      `<div style="padding:40px;font-family:system-ui;color:#e8eefc">
         <h2>Could not reach the backend</h2>
         <p>${error.message}</p>
         <p>Start it with:<br><code>uvicorn backend.main:app --reload</code></p>
       </div>`;
  }
})();
