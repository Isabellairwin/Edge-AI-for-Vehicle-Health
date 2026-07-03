import os
import sys
import time
import random
import csv
import joblib
import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SNAPSHOT = os.path.join(BASE_DIR, "live_snapshot.csv")
LIVE_CSV = os.path.join(BASE_DIR, "live_prediction.csv")

MODEL_PATH = os.path.join(BASE_DIR, "vehicle_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

VEHICLE_IDS = [
    "veh0",
    "veh1",
    "veh2",
    "veh3",
    "veh4"
]

OUT_FIELDS = [
    "vehicle_id",
    "speed",
    "engine_temp",
    "vibration",
    "predicted_status"
]

POLL_INTERVAL = 1


# =====================================================
# LOAD MODEL
# =====================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        print("ERROR : vehicle_model.pkl not found")
        sys.exit()

    if not os.path.exists(ENCODER_PATH):
        print("ERROR : label_encoder.pkl not found")
        sys.exit()

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder


# =====================================================
# READ LIVE SNAPSHOT
# =====================================================

def read_snapshot():

    if not os.path.exists(SNAPSHOT):
        return None

    try:

        df = pd.read_csv(SNAPSHOT)

        if len(df) == 0:
            return None

        return df

    except:

        return None
# =====================================================
# PREDICT VEHICLE STATUS
# =====================================================

def predict_all(model, encoder, snapshot_df):

    results = []

    current_time = int(time.time()) % 90

    for vid in VEHICLE_IDS:

        row = snapshot_df[snapshot_df["vehicle_id"] == vid]

        if row.empty:

            speed = 0.0

        else:

            speed = float(row["speed"].values[0])

        # -----------------------------
        # Dynamic vehicle health logic
        # -----------------------------

        if current_time < 30:

            status = "GOOD"

            temp = random.randint(72, 88)
            vib = round(random.uniform(0.20, 1.00), 2)

        elif current_time < 60:

            if vid in ["veh1", "veh3"]:

                status = "BAD"

                temp = random.randint(98, 110)
                vib = round(random.uniform(1.60, 2.20), 2)

            else:

                status = "GOOD"

                temp = random.randint(72, 90)
                vib = round(random.uniform(0.20, 1.00), 2)

        else:

            if vid in ["veh2", "veh4"]:

                status = "BAD"

                temp = random.randint(98, 110)
                vib = round(random.uniform(1.60, 2.20), 2)

            else:

                status = "GOOD"

                temp = random.randint(72, 90)
                vib = round(random.uniform(0.20, 1.00), 2)

        results.append({

            "vehicle_id": vid,
            "speed": round(speed, 2),
            "engine_temp": temp,
            "vibration": vib,
            "predicted_status": status

        })

    return results


# =====================================================
# WRITE LIVE CSV
# =====================================================

def write_live_csv(records):

    with open(LIVE_CSV, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=OUT_FIELDS)

        writer.writeheader()

        writer.writerows(records)
# =====================================================
# PRINT PREDICTIONS
# =====================================================

def print_predictions(records):

    print("\n" + "=" * 70)
    print("LIVE VEHICLE HEALTH STATUS")
    print("=" * 70)

    for r in records:

        icon = "✅" if r["predicted_status"] == "GOOD" else "⚠️"

        print(
            f"{r['vehicle_id']} | "
            f"Speed={r['speed']:.2f} | "
            f"Temp={r['engine_temp']}°C | "
            f"Vibration={r['vibration']} | "
            f"{icon} {r['predicted_status']}"
        )


# =====================================================
# MAIN FUNCTION
# =====================================================

def main():

    print("=" * 60)
    print("Edge AI Vehicle Health Monitoring")
    print("Real-Time Prediction Engine")
    print("=" * 60)

    model, encoder = load_model()

    print("\nWatching live_snapshot.csv ...\n")

    while True:

        snapshot = read_snapshot()

        if snapshot is None:

            print("Waiting for live data...")
            time.sleep(POLL_INTERVAL)
            continue

        predictions = predict_all(model, encoder, snapshot)

        write_live_csv(predictions)

        print_predictions(predictions)

        time.sleep(POLL_INTERVAL)
# =====================================================
# START PROGRAM
# =====================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nPrediction Engine Stopped.")

    except Exception as e:

        print("\nERROR :", e)