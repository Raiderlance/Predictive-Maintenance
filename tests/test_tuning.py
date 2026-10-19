from src.tune_hyperparameters import candidate_parameters


def test_candidate_parameters_starts_with_baseline_and_is_reproducible():
    first = candidate_parameters(4, random_state=42)
    second = candidate_parameters(4, random_state=42)

    assert first == second
    assert len(first) == 4
    assert first[0]["n_estimators"] == 300
    assert first[0]["learning_rate"] == 0.05
    assert first[0]["num_leaves"] == 31


def test_candidate_parameters_rejects_empty_search():
    try:
        candidate_parameters(0, random_state=42)
    except ValueError as error:
        assert "at least 1" in str(error)
    else:
        raise AssertionError("Expected an empty search to be rejected")
