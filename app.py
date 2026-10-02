import os
import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from models.gan_defect_generator import synthetic_gan, DEFECT_CLASSES
from models.tinyml_anomaly_detector import tinyml_inspector
from models.pinn_thermal_model import pinn_solver
from models.explainability import explainability_engine
from pipeline.iiot_simulator import iiot_pipeline
from pipeline.predictive_maintenance import maintenance_scheduler

app = FastAPI(
    title="TinyML IIoT & GAN Edge Anomaly Platform",
    description="Scalable IIoT Telemetry Ingestion, GAN Defect Synthesis, TinyML Edge Vision, PINN Physical Validation, and SHAP/LIME Predictive Maintenance.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schemas
class AnomalyInjectionRequest(BaseModel):
    mode: str # "NONE", "BEARING_OVERHEAT", "JOINT_TORQUE_SPIKE", "VIBRATION_FATIGUE"

class GANGenerateRequest(BaseModel):
    defect_type_idx: int = 0
    severity: float = 0.85
    seed: Optional[int] = None

class LatentInterpolateRequest(BaseModel):
    class_a: int = 0
    class_b: int = 1
    steps: int = 5

class CustomTelemetryRequest(BaseModel):
    feature_vector: List[float]

# Mount static folder
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/api/system/status")
async def get_system_status():
    return {
        "status": "ONLINE",
        "pipeline": "ACTIVE (OPC-UA, MQTT, Modbus-TCP -> InfluxDB)",
        "edge_accelerator": "ARM Cortex-M55 / Ethos-U55 INT8 Quantized",
        "gan_synthesizer": "cDCGAN Active",
        "pinn_engine": "Thermal-Stress PDE Autograd Active",
        "xai_engine": "SHAP TreeExplainer & LIME Surrogate Active",
        "total_points_ingested": iiot_pipeline.total_ingested_points
    }

@app.get("/api/telemetry/live")
async def get_live_telemetry():
    packet = iiot_pipeline.generate_raw_tick()
    return packet

@app.post("/api/telemetry/inject_anomaly")
async def inject_anomaly(req: AnomalyInjectionRequest):
    iiot_pipeline.set_anomaly_mode(req.mode)
    return {"status": "SUCCESS", "current_anomaly_mode": req.mode}

@app.get("/api/gan/classes")
async def get_defect_classes():
    return {"classes": DEFECT_CLASSES}

@app.post("/api/gan/generate")
async def generate_synthetic_defect(req: GANGenerateRequest):
    sample = synthetic_gan.generate_defect_sample(
        defect_type_idx=req.defect_type_idx,
        severity=req.severity,
        seed=req.seed
    )
    return sample

@app.post("/api/gan/interpolate")
async def interpolate_latent(req: LatentInterpolateRequest):
    interpolations = synthetic_gan.interpolate_latent_space(
        class_a=req.class_a,
        class_b=req.class_b,
        steps=req.steps
    )
    return {"interpolations": interpolations}

@app.post("/api/tinyml/inspect")
async def inspect_tinyml_vision(sample: Dict[str, Any]):
    result = tinyml_inspector.inspect_sample(sample)
    return result

@app.get("/api/pinn/evaluate")
async def evaluate_pinn(
    torque_nm: float = 42.5,
    speed_rpm: float = 1450.0,
    current_a: float = 14.2,
    sensor_temp_c: float = 68.4
):
    result = pinn_solver.evaluate_telemetry(
        torque_nm=torque_nm,
        speed_rpm=speed_rpm,
        current_a=current_a,
        sensor_temp_c=sensor_temp_c
    )
    return result

@app.post("/api/explainability/shap")
async def explain_shap(req: CustomTelemetryRequest):
    shap_res = explainability_engine.explain_with_shap(req.feature_vector)
    return shap_res

@app.post("/api/explainability/lime")
async def explain_lime(req: CustomTelemetryRequest):
    lime_res = explainability_engine.explain_with_lime(req.feature_vector)
    return lime_res

@app.get("/api/maintenance/schedule")
async def get_maintenance_schedule():
    packet = iiot_pipeline.generate_raw_tick()
    vec = packet["feature_vector"]
    sig = packet["signals"]
    
    shap_res = explainability_engine.explain_with_shap(vec)
    pinn_res = pinn_solver.evaluate_telemetry(
        torque_nm=sig["torque_nm"],
        speed_rpm=sig["motor_speed_rpm"],
        current_a=sig["drive_current_a"],
        sensor_temp_c=sig["bearing_temp_c"]
    )
    # Generate random test sample for vision edge
    gan_sample = synthetic_gan.generate_defect_sample(defect_type_idx=0, severity=0.1)
    tinyml_res = tinyml_inspector.inspect_sample(gan_sample)
    
    work_order = maintenance_scheduler.assess_health_and_schedule(
        signals=sig,
        shap_explanation=shap_res,
        pinn_eval=pinn_res,
        tinyml_vision=tinyml_res
    )
    return {
        "telemetry": packet,
        "shap_explanation": shap_res,
        "pinn_evaluation": pinn_res,
        "tinyml_vision": tinyml_res,
        "work_order": work_order
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            packet = iiot_pipeline.generate_raw_tick()
            vec = packet["feature_vector"]
            sig = packet["signals"]
            
            # Real-time PINN & Explainability eval on stream
            shap_res = explainability_engine.explain_with_shap(vec)
            pinn_res = pinn_solver.evaluate_telemetry(
                torque_nm=sig["torque_nm"],
                speed_rpm=sig["motor_speed_rpm"],
                current_a=sig["drive_current_a"],
                sensor_temp_c=sig["bearing_temp_c"]
            )
            
            payload = {
                "telemetry": packet,
                "pinn": pinn_res,
                "shap_brief": {
                    "degradation_score": shap_res["predicted_degradation_score"],
                    "top_drivers": shap_res["dominant_contributors"]
                }
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0) # 1Hz websocket stream update
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
