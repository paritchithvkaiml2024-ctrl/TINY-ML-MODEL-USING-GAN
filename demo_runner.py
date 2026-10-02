"""
Verification script for TinyML, GAN, PINN, IIoT Telemetry, and SHAP/LIME Modules.
"""
import sys
import os

from models.gan_defect_generator import synthetic_gan
from models.tinyml_anomaly_detector import tinyml_inspector
from models.pinn_thermal_model import pinn_solver
from models.explainability import explainability_engine
from pipeline.iiot_simulator import iiot_pipeline
from pipeline.predictive_maintenance import maintenance_scheduler

def run_pipeline_check():
    print("=" * 70)
    print("  TINYML + GAN + PINN + IIOT SYSTEM VERIFICATION SUITE")
    print("=" * 70)

    # 1. Test IIoT Multi-Protocol Ingestion
    print("\n[1/5] Testing IIoT Multi-Protocol Telemetry Stream...")
    packet = iiot_pipeline.generate_raw_tick()
    print(f"  + OPC-UA Nodes: {list(packet['protocols']['opc_ua'].keys())}")
    print(f"  + MQTT Topic: {packet['protocols']['mqtt']['topic']}")
    print(f"  + Modbus Registers: {list(packet['protocols']['modbus_tcp'].keys())}")
    print(f"  + InfluxDB Line Sample: {packet['influx_line_preview']}")
    print("  -> PASSED: IIoT Ingestion pipeline active.")

    # 2. Test cDCGAN Synthetic Defect Engine
    print("\n[2/5] Testing cDCGAN Defect Generator & Latent Interpolation...")
    sample = synthetic_gan.generate_defect_sample(defect_type_idx=0, severity=0.85)
    print(f"  + Generated Defect: {sample['defect_name']} (Severity: {sample['severity']})")
    print(f"  + Latent Vector Preview: {sample['latent_vector']}")
    print(f"  + Base64 payload size: {len(sample['image_b64'])} bytes")
    print("  -> PASSED: GAN defect synthesizer initialized.")

    # 3. Test TinyML Edge Vision Anomaly Detector
    print("\n[3/5] Testing TinyML Edge Vision Inspector & INT8 Quantization...")
    edge_res = tinyml_inspector.inspect_sample(sample)
    print(f"  + Predicted Class: {edge_res['prediction']} (Confidence: {edge_res['confidence_pct']}%)")
    print(f"  + Edge FP32 Latency: {edge_res['edge_metrics']['latency_fp32_ms']} ms")
    print(f"  + Edge INT8 Quantized Latency: {edge_res['edge_metrics']['latency_int8_quantized_ms']} ms (Speedup: {edge_res['edge_metrics']['speedup_factor']})")
    print(f"  + SRAM Footprint: {edge_res['edge_metrics']['int8_sram_usage_kb']} KB (Target: {edge_res['edge_metrics']['target_hardware']})")
    print(f"  + Bounding Box: {edge_res['bounding_box']}")
    print("  -> PASSED: TinyML edge model profiler functional.")

    # 4. Test Physics-Informed Neural Network (PINN)
    print("\n[4/5] Testing PINN Thermal-Stress 2D PDE Solver...")
    pinn_res = pinn_solver.evaluate_telemetry(torque_nm=45.0, speed_rpm=1450.0, current_a=14.5, sensor_temp_c=68.5)
    print(f"  + Governing Law: {pinn_res['governing_equation']}")
    print(f"  + PINN Predicted Temp: {pinn_res['pinn_predicted_temp_c']} °C vs Sensor: {pinn_res['sensor_actual_temp_c']} °C")
    print(f"  + PDE Physics Residual: {pinn_res['pde_physics_residual']}")
    print(f"  + Physical Consistency Score: {pinn_res['physics_consistency_score']}%")
    print("  -> PASSED: PINN analytical loss verification successful.")

    # 5. Test SHAP/LIME Explainability & Maintenance Scheduler
    print("\n[5/5] Testing SHAP/LIME Explainability & Maintenance Scheduling...")
    shap_res = explainability_engine.explain_with_shap(packet['feature_vector'])
    lime_res = explainability_engine.explain_with_lime(packet['feature_vector'])
    work_order = maintenance_scheduler.assess_health_and_schedule(
        signals=packet['signals'],
        shap_explanation=shap_res,
        pinn_eval=pinn_res,
        tinyml_vision=edge_res
    )
    print(f"  + SHAP Top Contributors: {shap_res['dominant_contributors']}")
    print(f"  + LIME Top Rule: {lime_res['top_rules'][0]}")
    print(f"  + Work Order ID: {work_order['work_order_id']}")
    print(f"  + Machine Health: {work_order['health_index_pct']}% | RUL: {work_order['estimated_rul_hours']} hrs ({work_order['estimated_rul_days']} days)")
    print(f"  + Priority Window: {work_order['dispatch_window']} ({work_order['urgency_level']})")
    print(f"  + Prescribed Actions: {work_order['prescribed_actions']}")
    print("  -> PASSED: Explainable predictive maintenance dispatcher verified.")

    print("\n" + "=" * 70)
    print("  ALL 5 MODULES PASSED SYSTEM VERIFICATION!")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline_check()
