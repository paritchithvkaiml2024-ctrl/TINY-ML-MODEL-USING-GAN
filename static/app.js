// TinyML IIoT & GAN Edge Anomaly Platform Frontend Engine

let telemetryChart;
const maxChartPoints = 25;
const chartTimestamps = [];
const dataTorque = [];
const dataTemp = [];
const dataVibration = [];
const dataCurrent = [];

let currentDefectIdx = 0;
let currentSeverity = 0.85;
let currentSample = null;

// Initialize when DOM ready
document.addEventListener("DOMContentLoaded", () => {
  initChart();
  generateGANSample();
  runMorphing();
  connectTelemetryStream();
});

// Initialize Chart.js Telemetry Monitor
function initChart() {
  const ctx = document.getElementById("telemetryChart").getContext("2d");
  
  telemetryChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: chartTimestamps,
      datasets: [
        {
          label: "Joint 3 Torque (Nm)",
          data: dataTorque,
          borderColor: "#00f0ff",
          backgroundColor: "rgba(0, 240, 255, 0.1)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: "Bearing Temp (°C)",
          data: dataTemp,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.1)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: "Vibration RMS (g × 10)",
          data: dataVibration,
          borderColor: "#ef4444",
          backgroundColor: "rgba(239, 68, 68, 0.1)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: "Drive Current (A)",
          data: dataCurrent,
          borderColor: "#8b5cf6",
          backgroundColor: "rgba(139, 92, 246, 0.1)",
          borderWidth: 1.5,
          pointRadius: 0,
          borderDash: [4, 4],
          tension: 0.35
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b", font: { size: 10 } }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b", font: { size: 10 } }
        }
      },
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { size: 11 }, boxWidth: 12 }
        }
      }
    }
  });
}

// WebSocket Live Telemetry Connection
function connectTelemetryStream() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
  let ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleTelemetryTick(data);
    } catch (e) {
      console.error("WS Parse error", e);
    }
  };

  ws.onerror = () => {
    // Fallback polling if WS closed
    setInterval(pollTelemetryFallback, 1500);
  };

  ws.onclose = () => {
    setTimeout(connectTelemetryStream, 3000);
  };
}

async function pollTelemetryFallback() {
  try {
    const res = await fetch("/api/maintenance/schedule");
    const data = await res.json();
    handleTelemetryTick({
      telemetry: data.telemetry,
      pinn: data.pinn_evaluation,
      shap_brief: {
        degradation_score: data.shap_explanation.predicted_degradation_score,
        top_drivers: data.shap_explanation.dominant_contributors
      }
    });
  } catch (e) {
    console.error("Polling error", e);
  }
}

// Handle Real-time Ingestion Tick
function handleTelemetryTick(payload) {
  const telem = payload.telemetry;
  const sig = telem.signals;
  const pinn = payload.pinn;

  // 1. Update Chart
  if (chartTimestamps.length >= maxChartPoints) {
    chartTimestamps.shift();
    dataTorque.shift();
    dataTemp.shift();
    dataVibration.shift();
    dataCurrent.shift();
  }

  chartTimestamps.push(telem.timestamp);
  dataTorque.push(sig.torque_nm);
  dataTemp.push(sig.bearing_temp_c);
  dataVibration.push(sig.vibration_rms_g * 10);
  dataCurrent.push(sig.drive_current_a);

  telemetryChart.update();

  // 2. Update KPI Ribbon
  document.getElementById("kpi-total-points").innerText = telem.pipeline_stats.total_points_ingested.toLocaleString();
  document.getElementById("kpi-latency").innerHTML = `${telem.pipeline_stats.ingestion_latency_ms} <span class="unit">ms</span>`;
  document.getElementById("kpi-physics-score").innerHTML = `${pinn.physics_consistency_score} <span class="unit">%</span>`;

  // 3. Update Influx Terminal Preview
  const terminal = document.getElementById("influx-terminal-output");
  terminal.innerText = `${telem.influx_line_preview}\n[OPC-UA] ${JSON.stringify(telem.protocols.opc_ua)}\n[MQTT] QoS:1 Topic: ${telem.protocols.mqtt.topic}`;

  // 4. Update PINN Spatial Heatmap & Values
  renderPINNHeatmap(pinn.spatial_heat_grid_10x10);
  document.getElementById("pinn-residual-val").innerText = pinn.pde_physics_residual;
  document.getElementById("pinn-temp-val").innerText = `${pinn.pinn_predicted_temp_c} °C`;
  document.getElementById("pinn-div-val").innerText = `${pinn.temp_divergence_c} °C`;
  
  const pinnBadge = document.getElementById("pinn-status-badge");
  if (pinn.temp_divergence_c > 15.0) {
    pinnBadge.className = "badge badge-danger";
    pinnBadge.innerText = "Physics Violation";
  } else if (pinn.temp_divergence_c > 6.0) {
    pinnBadge.className = "badge badge-warning";
    pinnBadge.innerText = "Thermal Drift";
  } else {
    pinnBadge.className = "badge badge-success";
    pinnBadge.innerText = "Physics Validated";
  }

  // 5. Update SHAP & Maintenance Work Orders periodically
  updateExplainabilityAndSchedule(telem.feature_vector);
}

// Render 10x10 Spatial Heatmap Grid
function renderPINNHeatmap(grid) {
  const container = document.getElementById("pinn-spatial-heatmap");
  if (!grid || !grid.length) return;
  
  container.innerHTML = "";
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[r].length; c++) {
      const temp = grid[r][c];
      const cell = document.createElement("div");
      cell.className = "grid-cell";
      
      // Calculate color between 25C (cyan/blue) and 100C (red/orange)
      const ratio = Math.max(0, Math.min(1, (temp - 25.0) / 75.0));
      const red = Math.round(ratio * 255);
      const blue = Math.round((1 - ratio) * 255);
      const green = Math.round(Math.sin(ratio * Math.PI) * 200);
      
      cell.style.backgroundColor = `rgb(${red}, ${green}, ${blue})`;
      cell.title = `Grid [${r},${c}]: ${temp}°C`;
      container.appendChild(cell);
    }
  }
}

// Trigger Anomaly Injection Modes
async function setAnomalyMode(mode) {
  try {
    await fetch("/api/telemetry/inject_anomaly", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode })
    });
  } catch (e) {
    console.error("Anomaly injection error", e);
  }
}

// GAN Defect Generation
async function generateGANSample() {
  try {
    const res = await fetch("/api/gan/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        defect_type_idx: parseInt(currentDefectIdx),
        severity: parseFloat(currentSeverity)
      })
    });
    const data = await res.json();
    currentSample = data;

    document.getElementById("gan-preview-img").src = data.image_b64;
    document.getElementById("gan-defect-title").innerText = data.defect_name;

    // Send to Edge Vision Inspector
    runEdgeInspection(data);
  } catch (e) {
    console.error("GAN Generation error", e);
  }
}

// Latent Space Morphing Strip
async function runMorphing() {
  try {
    const res = await fetch("/api/gan/interpolate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        class_a: 0,
        class_b: 1,
        steps: 6
      })
    });
    const data = await res.json();
    const container = document.getElementById("morph-strip-container");
    container.innerHTML = "";

    data.interpolations.forEach((item) => {
      const img = document.createElement("img");
      img.className = "morph-thumb";
      img.src = item.image_b64;
      img.title = `${item.defect_name} (Severity: ${Math.round(item.severity * 100)}%)`;
      img.onclick = () => {
        document.getElementById("gan-preview-img").src = item.image_b64;
        document.getElementById("gan-defect-title").innerText = item.defect_name;
        runEdgeInspection(item);
      };
      container.appendChild(img);
    });
  } catch (e) {
    console.error("Morphing error", e);
  }
}

// Run TinyML Edge Vision Inference
async function runEdgeInspection(sample) {
  try {
    const res = await fetch("/api/tinyml/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sample)
    });
    const data = await res.json();

    document.getElementById("edge-camera-img").src = sample.image_b64;
    document.getElementById("edge-fp32-lat").innerHTML = `${data.edge_metrics.latency_fp32_ms} <span class="unit">ms</span>`;
    document.getElementById("edge-int8-lat").innerHTML = `${data.edge_metrics.latency_int8_quantized_ms} <span class="unit">ms</span>`;
    document.getElementById("kpi-edge-lat").innerHTML = `${data.edge_metrics.latency_int8_quantized_ms} <span class="unit">ms</span>`;

    const edgeBadge = document.getElementById("badge-edge-status");
    const bboxOverlay = document.getElementById("bbox-overlay");

    if (data.is_anomaly && data.bounding_box) {
      edgeBadge.className = "badge badge-danger";
      edgeBadge.innerText = `Defect Detected (${data.confidence_pct}%)`;

      // Scale 64x64 coords to camera viewport
      const bbox = data.bounding_box;
      const scale = 220 / 64; // height scale
      bboxOverlay.style.display = "block";
      bboxOverlay.style.left = `${bbox.x_min * scale * 0.9}px`;
      bboxOverlay.style.top = `${bbox.y_min * scale}px`;
      bboxOverlay.style.width = `${(bbox.x_max - bbox.x_min) * scale * 0.9}px`;
      bboxOverlay.style.height = `${(bbox.y_max - bbox.y_min) * scale}px`;

      document.getElementById("bbox-label").innerText = `${bbox.defect_label}: ${bbox.confidence}%`;
    } else {
      edgeBadge.className = "badge badge-success";
      edgeBadge.innerText = "Nominal (0 Defects)";
      bboxOverlay.style.display = "none";
    }
  } catch (e) {
    console.error("Edge inspection error", e);
  }
}

// Update Explainability Bars and Work Order Dispatch
async function updateExplainabilityAndSchedule(featureVector) {
  try {
    const res = await fetch("/api/explainability/shap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feature_vector: featureVector })
    });
    const shapData = await res.json();

    // Render SHAP Bars
    const container = document.getElementById("shap-breakdown-list");
    container.innerHTML = "";

    shapData.feature_attributions.slice(0, 4).forEach((item) => {
      const isPos = item.shap_value > 0;
      const barItem = document.createElement("div");
      barItem.className = "xai-bar-item";
      barItem.innerHTML = `
        <div class="xai-bar-header">
          <span style="color:var(--text-secondary)">${item.feature}</span>
          <span style="font-family:var(--font-mono); color:${isPos ? 'var(--accent-rose)' : 'var(--accent-emerald)'}">
            ${isPos ? '+' : ''}${item.shap_value} (${item.importance_pct}%)
          </span>
        </div>
        <div class="xai-bar-track">
          <div class="xai-bar-fill ${isPos ? 'fill-positive' : 'fill-negative'}" style="width: ${Math.min(100, item.importance_pct * 2)}%"></div>
        </div>
      `;
      container.appendChild(barItem);
    });

    // Fetch Maintenance Work Order Schedule
    const schedRes = await fetch("/api/maintenance/schedule");
    const schedData = await schedRes.json();
    const wo = schedData.work_order;

    document.getElementById("kpi-health-index").innerHTML = `${wo.health_index_pct} <span class="unit">%</span>`;
    document.getElementById("kpi-rul").innerHTML = `${wo.estimated_rul_hours.toLocaleString()} <span class="unit">hrs</span>`;
    
    document.getElementById("wo-id").innerText = wo.work_order_id;
    document.getElementById("wo-window").innerText = wo.dispatch_window;
    document.getElementById("wo-window").style.color = wo.priority_color;

    const urgencyBadge = document.getElementById("workorder-urgency-badge");
    urgencyBadge.innerText = wo.urgency_level.split(" ")[0];
    urgencyBadge.style.backgroundColor = `${wo.priority_color}22`;
    urgencyBadge.style.color = wo.priority_color;
    urgencyBadge.style.borderColor = `${wo.priority_color}66`;

    const taskList = document.getElementById("wo-task-list");
    taskList.innerHTML = "";
    wo.prescribed_actions.forEach((task) => {
      const li = document.createElement("li");
      li.innerText = task;
      taskList.appendChild(li);
    });
  } catch (e) {
    console.error("Explainability update error", e);
  }
}

// UI Event Handlers
function onDefectTypeChange() {
  const select = document.getElementById("select-defect-type");
  currentDefectIdx = parseInt(select.value);
  document.getElementById("label-defect-type").innerText = select.options[select.selectedIndex].text;
  generateGANSample();
}

function onSeverityInput(val) {
  currentSeverity = parseFloat(val);
  document.getElementById("label-severity-val").innerText = `${Math.round(val * 100)}%`;
  generateGANSample();
}
