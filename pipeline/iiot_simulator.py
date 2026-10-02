import time
import math
import random
import json
from collections import deque
from typing import Dict, Any, List

class IIoTProtocolTelemetryPipeline:
    """
    High-Frequency IIoT Multi-Protocol Telemetry Ingestion Pipeline.
    Simulates industrial ingestion from OPC-UA, MQTT, and Modbus-TCP endpoints
    streaming into an InfluxDB-compatible time-series buffer at enterprise scale.
    """
    def __init__(self, buffer_size=500):
        self.buffer_size = buffer_size
        self.telemetry_history = deque(maxlen=buffer_size)
        self.influx_line_buffer = deque(maxlen=buffer_size)
        self.total_ingested_points = 24_850_000 # Pre-seeded realistic daily millions count
        self.start_time = time.time()
        
        # Robot kinematic baseline states
        self.robot_id = "ROBOT_ARM_KR210_R2700"
        self.workcell_id = "AUTOMATION_CELL_04"
        self.anomaly_injection_mode = "NONE" # "NONE", "BEARING_OVERHEAT", "JOINT_TORQUE_SPIKE", "VIBRATION_FATIGUE"

    def set_anomaly_mode(self, mode: str):
        self.anomaly_injection_mode = mode

    def generate_raw_tick(self) -> Dict[str, Any]:
        """
        Generates multi-protocol high-frequency sensor readings from the robotic arm
        and thermal camera array.
        """
        t = time.time() - self.start_time
        self.total_ingested_points += 48 # ~48 multi-channel sensor parameters per tick
        
        # Base kinematic motion dynamics
        joint_angle_deg = 45.0 + 35.0 * math.sin(t * 0.8)
        joint_velocity = 35.0 * 0.8 * math.cos(t * 0.8)
        
        # Normal baseline telemetry values
        torque_nm = 38.0 + 8.5 * abs(math.sin(t * 1.2)) + random.gauss(0, 0.8)
        motor_speed_rpm = 1450.0 + 120.0 * math.sin(t * 0.8) + random.gauss(0, 5.0)
        drive_current_a = 12.5 + (torque_nm / 10.0) * 1.8 + random.gauss(0, 0.3)
        bearing_temp_c = 58.0 + 6.0 * math.sin(t * 0.1) + random.gauss(0, 0.4)
        vibration_rms_g = 1.2 + 0.3 * abs(math.sin(t * 2.5)) + random.gauss(0, 0.05)
        inverter_jitter_hz = 0.18 + random.uniform(0.02, 0.08)
        acoustic_emission_db = 52.0 + 4.0 * math.sin(t * 0.5) + random.gauss(0, 0.5)
        thermal_gradient = 0.12 + random.uniform(0.01, 0.05)
        kinematic_error_mm = 0.08 + random.uniform(0.01, 0.04)

        # Apply anomaly injections if triggered
        if self.anomaly_injection_mode == "BEARING_OVERHEAT":
            bearing_temp_c += 32.5 + random.uniform(2, 6) # > 90°C
            thermal_gradient += 1.85
            vibration_rms_g += 1.8
        elif self.anomaly_injection_mode == "JOINT_TORQUE_SPIKE":
            torque_nm += 45.0 + random.uniform(5, 12) # High mechanical overload
            drive_current_a += 18.0
            kinematic_error_mm += 0.85
        elif self.anomaly_injection_mode == "VIBRATION_FATIGUE":
            vibration_rms_g += 4.8 + random.uniform(0.5, 1.2) # Severe mechanical resonance
            acoustic_emission_db += 28.0
            inverter_jitter_hz += 1.2

        # 1. OPC-UA Protocol Payload Representation
        opc_ua_nodes = {
            "ns=2;s=Robotics.KR210.Joint3.ActualTorque": round(torque_nm, 2),
            "ns=2;s=Robotics.KR210.Joint3.PositionDeg": round(joint_angle_deg, 2),
            "ns=2;s=Robotics.KR210.Joint3.VelocityDegS": round(joint_velocity, 2),
            "ns=2;s=Robotics.KR210.Sensors.ThermalArray.MaxC": round(bearing_temp_c, 2)
        }

        # 2. MQTT Telemetry Topic Payload
        mqtt_packet = {
            "topic": f"factory/{self.workcell_id}/{self.robot_id}/telemetry",
            "qos": 1,
            "payload": {
                "timestamp_ns": int(time.time() * 1e9),
                "vibration_rms": round(vibration_rms_g, 3),
                "acoustic_db": round(acoustic_emission_db, 1),
                "kinematic_err_mm": round(kinematic_error_mm, 3)
            }
        }

        # 3. Modbus-TCP Register Map Payload
        modbus_registers = {
            "0x3001_MotorRPM": int(motor_speed_rpm),
            "0x3002_DriveCurrent_mA": int(drive_current_a * 1000),
            "0x3003_InverterFreq_dHz": int(inverter_jitter_hz * 100),
            "0x3004_IGBTTemp_dC": int(bearing_temp_c * 10)
        }

        # Vector format for downstream SHAP / LIME and ML models
        feature_vector = [
            round(torque_nm, 2),
            round(vibration_rms_g, 2),
            round(bearing_temp_c, 2),
            round(drive_current_a, 2),
            round(inverter_jitter_hz, 2),
            round(acoustic_emission_db, 2),
            round(thermal_gradient, 2),
            round(kinematic_error_mm, 2)
        ]

        # InfluxDB Line Protocol Formatter
        timestamp_ns = int(time.time() * 1e9)
        influx_line = (
            f"robot_telemetry,cell={self.workcell_id},robot={self.robot_id} "
            f"torque={torque_nm:.2f},temp={bearing_temp_c:.2f},vibration={vibration_rms_g:.3f},"
            f"current={drive_current_a:.2f},rpm={motor_speed_rpm:.1f} {timestamp_ns}"
        )
        self.influx_line_buffer.append(influx_line)

        packet = {
            "timestamp": time.strftime("%H:%M:%S"),
            "timestamp_epoch": time.time(),
            "robot_id": self.robot_id,
            "workcell": self.workcell_id,
            "protocols": {
                "opc_ua": opc_ua_nodes,
                "mqtt": mqtt_packet,
                "modbus_tcp": modbus_registers
            },
            "signals": {
                "torque_nm": round(torque_nm, 2),
                "motor_speed_rpm": round(motor_speed_rpm, 1),
                "drive_current_a": round(drive_current_a, 2),
                "bearing_temp_c": round(bearing_temp_c, 2),
                "vibration_rms_g": round(vibration_rms_g, 3),
                "inverter_jitter_hz": round(inverter_jitter_hz, 3),
                "acoustic_emission_db": round(acoustic_emission_db, 1),
                "thermal_gradient": round(thermal_gradient, 3),
                "kinematic_error_mm": round(kinematic_error_mm, 3)
            },
            "feature_vector": feature_vector,
            "influx_line_preview": influx_line,
            "pipeline_stats": {
                "total_points_ingested": self.total_ingested_points,
                "daily_throughput_est": "3.84 Million pts/day",
                "ingestion_latency_ms": round(random.uniform(0.8, 2.4), 2),
                "buffer_fill_pct": round((len(self.telemetry_history) / self.buffer_size) * 100, 1),
                "active_protocols": ["OPC-UA (Binary)", "MQTT v5 (JSON)", "Modbus-TCP (Port 502)"]
            }
        }

        self.telemetry_history.append(packet)
        return packet

    def get_recent_history(self, limit=30) -> List[Dict[str, Any]]:
        return list(self.telemetry_history)[-limit:]

iiot_pipeline = IIoTProtocolTelemetryPipeline()
