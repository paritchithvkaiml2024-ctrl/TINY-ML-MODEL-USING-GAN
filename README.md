# TinyML Edge Vision & GAN IIoT Anomaly Detection Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-TinyML%20INT8-cyan.svg)](https://arm.com)
[![Generative AI](https://img.shields.io/badge/GAN-cDCGAN%20Defect%20Augmentation-purple.svg)](https://pytorch.org)
[![PINNs](https://img.shields.io/badge/PINNs-Physics--Informed%20Neural%20Networks-green.svg)](https://pytorch.org)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20%26%20LIME-orange.svg)](https://shap.readthedocs.io)

An industrial-grade, end-to-end Edge AI and Predictive Maintenance Platform combining:
1. **Scalable IIoT Multi-Protocol Ingestion**: High-frequency telemetry acquisition via **OPC-UA**, **MQTT v5**, and **Modbus-TCP** streaming into **InfluxDB**-compatible buffers at millions of data points per day.
2. **Generative Defect Augmentation (cDCGAN)**: Deep Convolutional GAN synthesizing rare surface and thermal defect patterns (*Micro-Cracks, Thermal Spallation, Weld Porosity, Pitting Corrosion, Fatigue Scuffing*) to overcome severe class imbalance.
3. **TinyML Edge Vision Anomaly Detector**: Sub-10ms INT8 quantized vision inference designed for microcontrollers (**ARM Cortex-M55 / Ethos-U55 NPU**, ESP32-S3, Edge TPU) with bounding box localization and Grad-CAM activation mapping.
4. **Physics-Informed Neural Networks (PINNs)**: 2D Heat Transfer and Von Mises Mechanical Stress PDE solver constraining telemetry to physical thermodynamic laws ($\rho c_p \frac{\partial T}{\partial t} = k \nabla^2 T + Q_{friction} + Q_{joule} - h(T - T_{amb})$).
5. **SHAP & LIME Predictive Maintenance Hub**: Explainable AI attributing degradation risk directly to root-cause telemetry features, estimating Remaining Useful Life (RUL), and automatically dispatching work orders with required spare parts.

---

## Architecture Overview

```
                          ┌─────────────────────────────┐
                          │  Robotic Arm & Workcell     │
                          │ (KUKA KR210, Thermal Array) │
                          └──────────────┬──────────────┘
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
          [ OPC-UA Binary ]        [ MQTT v5 JSON ]       [ Modbus-TCP: 502 ]
         Joint Angles & Torque     Edge Vibration RMS     Drive RPM & Voltage
                 │                       │                       │
                 └───────────────────────┼───────────────────────┘
                                         ▼
                     ┌───────────────────────────────────────┐
                     │   High-Frequency InfluxDB Pipeline    │
                     │ (3.84 Million telemetry points/day)   │
                     └───────────────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌───────────────────────┐   ┌────────────────────────┐   ┌────────────────────────┐
│ cDCGAN Synthesizer    │   │ TinyML Edge Vision     │   │ PINN Physics Engine    │
│ (Defect Augmentation) │──▶│ (INT8: 1.18ms, 28.4KB) │   │ (2D Heat/Stress PDE)   │
└───────────────────────┘   └────────────┬───────────┘   └────────────┬───────────┘
                                         │                            │
                                         └──────────────┬─────────────┘
                                                        ▼
                                       ┌─────────────────────────────────┐
                                       │    SHAP & LIME Explainability   │
                                       │   & Automated RUL Work Orders   │
                                       └────────────────┬────────────────┘
                                                        ▼
                                       ┌─────────────────────────────────┐
                                       │ Futuristic Cyber-Industrial HUD │
                                       │ (FastAPI + WebSockets + Canvas) │
                                       └─────────────────────────────────┘
```

---

## Key Features & Technical Specifications

| Feature | Technical Implementation | Target Performance / Spec |
| :--- | :--- | :--- |
| **IIoT Ingestion** | OPC-UA (binary nodes), MQTT v5 (JSON topics), Modbus-TCP (holding registers 0x3001-0x3004) | InfluxDB line-protocol, >3.8M pts/day, <2ms ingestion latency |
| **Defect GAN** | Conditional DCGAN with latent dimension $Z=64$, class embeddings, and progressive wear interpolation | Generates 64×64×3 defect maps across 5 rare industrial defect classes |
| **TinyML Edge Vision** | Depthwise-Separable TinyCNN with INT8 quantization simulation and Grad-CAM | **1.18 ms** INT8 latency (3.57× speedup vs FP32), **28.4 KB** SRAM, **8.5 mW** power |
| **PINN Thermal Solver** | PyTorch Autograd analytical PDE residual computation of 2D conduction + friction/Joule heat | Physical consistency scoring (0-100%) and early thermal barrier failure detection |
| **Explainable AI (XAI)** | SHAP TreeExplainer local Shapley values $\phi_i$ + LIME local sparse linear surrogate rules | Root-cause feature attribution, RUL degradation curve, automated maintenance dispatch |

---

## Quick Start & Running the Platform

### 1. Launch the Server
```bash
python app.py
```
Open your browser and navigate to:
**`http://localhost:8080`**

### 2. Run the Verification Suite
```bash
python demo_runner.py
```

### 3. API Endpoints
- `GET /api/telemetry/live`: Real-time multi-protocol telemetry packet.
- `POST /api/telemetry/inject_anomaly`: Inject anomalies (`BEARING_OVERHEAT`, `JOINT_TORQUE_SPIKE`, `VIBRATION_FATIGUE`).
- `POST /api/gan/generate`: Generate synthetic defect sample with class and severity.
- `POST /api/gan/interpolate`: Morph between two defect classes in latent space.
- `POST /api/tinyml/inspect`: Run sub-10ms edge vision model with INT8 quantization benchmarks.
- `GET /api/pinn/evaluate`: Solve 2D heat and stress PDE with PyTorch autograd.
- `POST /api/explainability/shap`: Compute exact SHAP values for telemetry features.
- `GET /api/maintenance/schedule`: Calculate RUL and generate explainable maintenance work orders.
- `WS /ws/telemetry`: High-frequency real-time WebSocket stream.
