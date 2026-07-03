"""
Edge AI Vehicle Health Monitoring System
main.py — SUMO Simulation + Training Data Collector

HOW TO RUN:
    python main.py

WHAT THIS DOES:
    1.  Opens SUMO-GUI (paused — press Play button to start)
    2.  Runs for exactly 99 simulation steps
    3.  Every step: reads speed for all 5 vehicles,
        generates engine_temp and vibration,
        prints a formatted status block to the terminal
    4.  Writes every row to vehicle_health.csv  (full history)
    5.  Also overwrites live_snapshot.csv every step with the
        latest 5 rows only — used by predict_live.py

PIPELINE:
    main.py  →  vehicle_health.csv  +  live_snapshot.csv
                            ↓
                    predict_live.py  →  live_prediction.csv
                                                ↓
                                        dashboard.py
"""

import os
import csv
import time
import random
import traci

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SUMO_HOME   = r"C:\Program Files (x86)\Eclipse\Sumo"
SUMO_EXE    = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")
SUMO_CFG    = os.path.join(BASE_DIR, "simulation.sumocfg")
HEALTH_CSV  = os.path.join(BASE_DIR, "vehicle_health.csv")
SNAPSHOT    = os.path.join(BASE_DIR, "live_snapshot.csv")

VEHICLE_IDS = ["veh0", "veh1", "veh2", "veh3", "veh4"]
SIM_STEPS   = 99        # exactly 99 seconds
STEP_DELAY  = 0.5       # 0.5 s between steps → matches 500 ms GUI delay

os.environ["SUMO_HOME"] = SUMO_HOME

CSV_FIELDS  = ["vehicle_id", "speed", "engine_temp", "vibration", "health_status"]

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def rule_based_health(engine_temp: int, vibration: float) -> str:
    """
    Rule used ONLY for generating training labels in vehicle_health.csv.
    predict_live.py replaces this with the actual ML model.
    """
    return "BAD" if engine_temp > 95 or vibration > 1.5 else "GOOD"


def write_snapshot(rows: list[dict]):
    """
    Overwrite live_snapshot.csv with exactly 5 rows (latest readings).
    predict_live.py watches this file to make real-time predictions.
    """
    with open(SNAPSHOT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_step(step: int, rows: list[dict]):
    """Print a clean formatted block for every vehicle in this step."""
    print(f"\n{'='*54}")
    print(f"  Step {step}  (Simulation second {step + 1} / {SIM_STEPS})")
    print(f"{'='*54}")
    for r in rows:
        status_icon = "✅" if r["health_status"] == "GOOD" else "⚠️ "
        print(
            f"  {r['vehicle_id']}  |  Speed: {r['speed']:5.2f} m/s  |  "
            f"Temp: {r['engine_temp']:3d} °C  |  "
            f"Vibration: {r['vibration']:.2f}  |  "
            f"Status: {status_icon}  {r['health_status']}"
        )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 54)
    print("  Edge AI Vehicle Health Monitoring System")
    print("  main.py — SUMO Training-Data Collector")
    print("=" * 54)
    print()
    print("  ➜  SUMO-GUI is opening ...")
    print("  ➜  Press the  ▶ Play  button in SUMO to start.")
    print()

    # ── Start SUMO-GUI (no --start → waits for user to press Play)
    # ── --delay 500 sets the GUI animation slider to 500 ms
    sumo_cmd = [
        SUMO_EXE,
        "-c", SUMO_CFG,
        "--delay", "500",          # 500 ms GUI delay per step (visible movement)
        "--quit-on-end", "false",  # SUMO stays open after simulation ends
    ]
    traci.start(sumo_cmd, label="main_sim")

    # ── Prepare vehicle_health.csv (write header once, then append every step)
    health_file = open(HEALTH_CSV, "w", newline="", encoding="utf-8")
    health_writer = csv.DictWriter(health_file, fieldnames=CSV_FIELDS)
    health_writer.writeheader()

    print(f"  ➜  Writing training data  → {HEALTH_CSV}")
    print(f"  ➜  Writing live snapshot  → {SNAPSHOT}")
    print(f"  ➜  Simulation will run for {SIM_STEPS} steps.\n")

    try:
        for step in range(SIM_STEPS):

            # ── Advance exactly ONE simulation step ──────────────────────
            traci.simulationStep()

            active = set(traci.vehicle.getIDList())
            step_rows = []

            for vid in VEHICLE_IDS:
                # Speed from SUMO (0.0 if vehicle not yet spawned)
                speed = round(traci.vehicle.getSpeed(vid), 2) if vid in active else 0.0

                # Simulated sensor readings
                engine_temp = random.randint(70, 110)
                vibration   = round(random.uniform(0.1, 2.0), 2)
                health      = rule_based_health(engine_temp, vibration)

                row = {
                    "vehicle_id"   : vid,
                    "speed"        : speed,
                    "engine_temp"  : engine_temp,
                    "vibration"    : vibration,
                    "health_status": health,
                }
                step_rows.append(row)

            # ── Terminal output ──────────────────────────────────────────
            print_step(step, step_rows)

            # ── Write to vehicle_health.csv (all history) ────────────────
            health_writer.writerows(step_rows)
            health_file.flush()   # flush so predict_live.py can read it immediately

            # ── Write to live_snapshot.csv (current state only) ──────────
            write_snapshot(step_rows)

            # ── Pace the loop to match 500 ms GUI delay ──────────────────
            time.sleep(STEP_DELAY)

    except traci.exceptions.FatalTraCIError:
        print("\n[INFO] SUMO window was closed — simulation stopped.")

    finally:
        health_file.close()
        try:
            traci.close()
        except Exception:
            pass

    print(f"\n{'='*54}")
    print(f"  Simulation complete — {SIM_STEPS} steps finished.")
    print(f"  vehicle_health.csv  → {len(VEHICLE_IDS) * SIM_STEPS} rows saved")
    print(f"  Next step: python train_model.py")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()