import pandas as pd

from f1ml.safety_car import add_prior_circuit_safety_features, build_safety_car_labels


def test_safety_car_labels_distinguish_deployment_from_ending_and_shift_history():
    sessions = pd.DataFrame(
        {
            "season": [2025, 2025, 2025],
            "round": [1, 2, 3],
            "session_key": [101, 102, 103],
            "circuit_key": [7, 7, 7],
        }
    )
    control = pd.DataFrame(
        {
            "session_key": [101, 102, 102, 103],
            "category": ["SafetyCar"] * 4,
            "message": [
                "VIRTUAL SAFETY CAR DEPLOYED",
                "SAFETY CAR DEPLOYED",
                "SAFETY CAR ENDING",
                "VSC ENDING",
            ],
        }
    )
    calendar = pd.DataFrame(
        {
            "season": [2025] * 3,
            "round": [1, 2, 3],
            "circuit_id": ["track"] * 3,
        }
    )
    labels = build_safety_car_labels(sessions, control, calendar)
    assert labels[["has_sc", "has_vsc"]].values.tolist() == [[0, 1], [1, 0], [0, 0]]

    features = add_prior_circuit_safety_features(calendar, labels)
    assert features.loc[0, "circuit_sc_rate_prior"] != features.loc[0, "circuit_sc_rate_prior"]
    assert features["circuit_sc_rate_prior"].tolist()[1:] == [0.0, 0.5]
    assert features["circuit_vsc_rate_prior"].tolist()[1:] == [1.0, 0.5]
