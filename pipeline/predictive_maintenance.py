import time
from typing import Dict, Any, List

class PredictiveMaintenanceScheduler:
    """
    RUL (Remaining Useful Life) Estimator and Automated Maintenance Dispatcher.
    Synthesizes multi-sensor degradation scores, PINN physics residuals, and SHAP feature
    attributions to generate automated, explainable maintenance work orders.
    """
    def __init__(self):
        self.nominal_lifespan_hours = 12000.0 # Standard robotic actuator bearing life
        
    def assess_health_and_schedule(
        self,
        signals: Dict[str, float],
        shap_explanation: Dict[str, Any],
        pinn_eval: Dict[str, Any],
        tinyml_vision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Integrates telemetry signals, vision inspection, and explainability vectors
        to compute RUL and schedule maintenance intervention.
        """
        deg_score = float(shap_explanation.get("predicted_degradation_score", 15.0))
        phys_score = float(pinn_eval.get("physics_consistency_score", 95.0))
        is_vision_defect = bool(tinyml_vision.get("is_anomaly", False))
        
        # Composite Health Index (0 - 100%)
        # 100 = Brand new pristine state, 0 = Catastrophic failure
        health_index = 100.0 - deg_score * 0.7 - (100.0 - phys_score) * 0.3
        if is_vision_defect:
            health_index -= 15.0
        health_index = max(2.0, min(99.5, health_index))
        
        # Calculate Remaining Useful Life (RUL)
        # Exponential wear model
        wear_rate = max(0.01, (100.0 - health_index) / 25.0)
        estimated_rul_hours = round(max(8.0, (health_index / 100.0) * self.nominal_lifespan_hours / wear_rate), 1)
        estimated_rul_days = round(estimated_rul_hours / 24.0, 1)
        
        # Determine Severity and Action Schedule
        if health_index < 40.0 or deg_score > 70.0:
            urgency = "CRITICAL (Immediate Action Required)"
            time_window = "Within 4 Hours"
            priority_color = "#ef4444"
        elif health_index < 65.0 or deg_score > 45.0:
            urgency = "HIGH (Scheduled Intervention)"
            time_window = "Within 24 Hours"
            priority_color = "#f59e0b"
        elif health_index < 85.0:
            urgency = "MODERATE (Standard Periodic Care)"
            time_window = "Next Scheduled Shift (7 Days)"
            priority_color = "#3b82f6"
        else:
            urgency = "OPTIMAL (Nominal Operation)"
            time_window = "Routine Maintenance in 45 Days"
            priority_color = "#10b981"

        # Generate root-cause informed action items from SHAP top contributors
        dominant = shap_explanation.get("dominant_contributors", [])
        recommended_actions = []
        parts_list = []
        
        top_feature_names = [f["feature"] for f in shap_explanation.get("feature_attributions", [])[:3]]
        
        if any("Bearing Temp" in f or "Thermal" in f for f in top_feature_names):
            recommended_actions.append("Flush & replenish high-viscosity synthetic grease in Joint 3 gearbox.")
            parts_list.append("KUKA Gear Lubricant Synthetic Polyurea 1.5L")
            
        if any("Vibration" in f or "Kinematic" in f for f in top_feature_names):
            recommended_actions.append("Inspect harmonic drive flexspline and check radial runout calibration.")
            parts_list.append("Precision Harmonic Drive Bearing Seal Kit")
            
        if any("Torque" in f or "Current" in f for f in top_feature_names):
            recommended_actions.append("Calibrate servo driver current loop and verify motor brake clearance.")
            parts_list.append("Brushless AC Servo Driver PCB (KR-C4)")
            
        if is_vision_defect:
            defect_name = tinyml_vision.get("prediction", "Surface Defect")
            recommended_actions.append(f"Visual QA edge trigger: Polish or replace surface segment showing '{defect_name}'.")

        if not recommended_actions:
            recommended_actions.append("Continuous condition monitoring. No immediate mechanical disassembly required.")
            parts_list.append("N/A - Preventative baseline monitoring")

        work_order = {
            "work_order_id": f"WO-IIOT-{int(time.time()) % 100000:05d}",
            "machine_id": "KR210_R2700_ROBOT_04",
            "health_index_pct": round(health_index, 1),
            "estimated_rul_hours": estimated_rul_hours,
            "estimated_rul_days": estimated_rul_days,
            "urgency_level": urgency,
            "dispatch_window": time_window,
            "priority_color": priority_color,
            "pinn_physics_consistency": round(phys_score, 1),
            "shap_primary_driver": dominant[0] if dominant else "Normal Wear",
            "prescribed_actions": recommended_actions,
            "required_spare_parts": parts_list,
            "technician_notes": (
                f"Generated automatically by SHAP/LIME telemetry engine. "
                f"Predicted Degradation Score: {deg_score:.1f}/100. "
                f"Physics Residual: {pinn_eval.get('pde_physics_residual', 0):.4f}."
            )
        }

        return work_order

maintenance_scheduler = PredictiveMaintenanceScheduler()
