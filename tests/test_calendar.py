import pandas as pd

from src.calendar_ import add_calendar_columns, phase_for_minute


def test_calendar() -> None:
    # 18:00 New York maps to different UTC hours across the DST transition,
    # while the exchange-local phase and session convention remain unchanged.
    frame = pd.DataFrame(
        {
            "root": ["ES", "ES", "ES", "HSI", "HSI"],
            "timestamp": pd.to_datetime(
                [
                    "2026-03-01 23:00:00Z",  # 18:00 EST, Sunday -> Monday session
                    "2026-03-08 22:00:00Z",  # 18:00 EDT, Sunday -> Monday session
                    "2026-03-09 13:30:00Z",  # 09:30 EDT
                    "2026-07-27 09:00:00Z",  # 17:00 HKT
                    "2026-07-27 17:00:00Z",  # 01:00 next calendar day HKT
                ],
                utc=True,
            ),
        }
    )

    result = add_calendar_columns(frame)

    assert result.loc[0, "local_minute"] == 18 * 60
    assert result.loc[1, "local_minute"] == 18 * 60
    assert str(result.loc[0, "session_id"]) == "2026-03-02"
    assert str(result.loc[1, "session_id"]) == "2026-03-09"
    assert result.loc[2, "session_phase"] == "US_OPEN"
    assert str(result.loc[3, "session_id"]) == "2026-07-27"
    assert str(result.loc[4, "session_id"]) == "2026-07-27"
    assert result.loc[4, "session_phase"] == "HK_LATE"


def test_phase_boundaries_are_left_closed() -> None:
    assert phase_for_minute("ES", 9 * 60 + 29) == "US_PRE"
    assert phase_for_minute("ES", 9 * 60 + 30) == "US_OPEN"
    assert phase_for_minute("HSI", 21 * 60 + 29) == "HK_EU"
    assert phase_for_minute("HSI", 21 * 60 + 30) == "HK_US_PRE"

