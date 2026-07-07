import pandas as pd
import os
from datetime import datetime


class SessionLogger:

    def __init__(self):

        self.records = []

    def add_record(
        self,
        timestamp,
        left_knee,
        right_knee,
        asymmetry,
        velocity,
        risk
    ):

        self.records.append({

            "timestamp": timestamp,

            "left_knee": round(left_knee, 2),

            "right_knee": round(right_knee, 2),

            "asymmetry": round(asymmetry, 2),

            "velocity": round(velocity, 2),

            "risk": round(risk,2),
            "risk_level": (
                "VERY HIGH" if risk>=75 else
                "HIGH" if risk>=50 else
                "MODERATE" if risk>=25 else
                "LOW"
            )

        })

    def get_record_count(self):

        return len(self.records)

    def clear(self):

        self.records = []

    def save_csv(self):

        if len(self.records) == 0:

            print("No records to save.")

            return None

        if not os.path.exists("sessions"):

            os.makedirs("sessions")

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = f"sessions/session_{timestamp}.csv"

        df = pd.DataFrame(
            self.records
        )

        df.to_csv(
            filename,
            index=False
        )

        print(
            f"Saved {len(self.records)} rows"
        )

        print(
            f"Location: {filename}"
        )

        return filename