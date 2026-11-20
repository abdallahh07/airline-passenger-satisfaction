from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from data_processing.preprocess import imputer
from model.predict import load_model
from utils.config import FEATURED_RAW

app = FastAPI(title="Airline Passenger Satisfaction API")
model = load_model()          # loaded once, when the server starts

Rating = Field(ge=0, le=5)    # every service rating is 0-5


class Passenger(BaseModel):
    gender: Literal["Male", "Female"]
    customer_type: Literal["Loyal Customer", "disloyal Customer"]
    age: int = Field(ge=0, le=120)
    type_of_travel: Literal["Business travel", "Personal Travel"]
    travel_class: Literal["Business", "Eco", "Eco Plus"]
    flight_distance: int = Field(ge=0)
    inflight_wifi_service: int = Rating
    departure_arrival_time_convenient: int = Rating
    ease_of_online_booking: int = Rating
    gate_location: int = Rating
    food_and_drink: int = Rating
    online_boarding: int = Rating
    seat_comfort: int = Rating
    inflight_entertainment: int = Rating
    on_board_service: int = Rating
    leg_room_service: int = Rating
    baggage_handling: int = Rating
    checkin_service: int = Rating
    cleanliness: int = Rating
    departure_delay_in_minutes: int = Field(ge=0)
    arrival_delay_in_minutes: float | None = Field(default=None, ge=0)


# API names (snake_case) -> the column names the model was trained on
COLUMN_MAP = {
    "gender": "Gender",
    "customer_type": "Customer Type",
    "age": "Age",
    "type_of_travel": "Type of Travel",
    "travel_class": "Class",
    "flight_distance": "Flight Distance",
    "inflight_wifi_service": "Inflight wifi service",
    "departure_arrival_time_convenient": "Departure/Arrival time convenient",
    "ease_of_online_booking": "Ease of Online booking",
    "gate_location": "Gate location",
    "food_and_drink": "Food and drink",
    "online_boarding": "Online boarding",
    "seat_comfort": "Seat comfort",
    "inflight_entertainment": "Inflight entertainment",
    "on_board_service": "On-board service",
    "leg_room_service": "Leg room service",
    "baggage_handling": "Baggage handling",
    "checkin_service": "Checkin service",
    "cleanliness": "Cleanliness",
    "departure_delay_in_minutes": "Departure Delay in Minutes",
    "arrival_delay_in_minutes": "Arrival Delay in Minutes",
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(passenger: Passenger):
    row = pd.DataFrame([passenger.model_dump()]).rename(columns=COLUMN_MAP)
    row["Arrival Delay in Minutes"] = row["Arrival Delay in Minutes"].astype(float)  # None -> NaN
    row = imputer(row)[FEATURED_RAW]
    prob = float(model.predict_proba(row)[:, 1][0])
    return {
        "satisfaction_probability": round(prob, 4),
        "prediction": "satisfied" if prob >= 0.5 else "neutral or dissatisfied",
    }