"""
§57 — OHLC validation and duplicate detection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.database.models import DataQualityStatus
from src.market.data_quality import validate_ohlc_bar


class TestOHLCValidation:
    def test_valid_bar(self):
        result = validate_ohlc_bar(open_=100, high=105, low=98, close=103, volume=10000)
        assert result.status == DataQualityStatus.VALID
        assert result.is_usable

    def test_high_less_than_low_is_invalid(self):
        result = validate_ohlc_bar(open_=100, high=90, low=95, close=92, volume=1000)
        assert result.status == DataQualityStatus.INVALID
        assert not result.is_usable
        assert any("high" in i and "low" in i for i in result.issues)

    def test_high_less_than_open_is_invalid(self):
        result = validate_ohlc_bar(open_=110, high=105, low=100, close=103, volume=1000)
        assert result.status == DataQualityStatus.INVALID

    def test_high_less_than_close_is_invalid(self):
        result = validate_ohlc_bar(open_=100, high=105, low=98, close=110, volume=1000)
        assert result.status == DataQualityStatus.INVALID

    def test_low_greater_than_open_is_invalid(self):
        result = validate_ohlc_bar(open_=90, high=105, low=95, close=100, volume=1000)
        assert result.status == DataQualityStatus.INVALID

    def test_low_greater_than_close_is_invalid(self):
        result = validate_ohlc_bar(open_=100, high=105, low=102, close=95, volume=1000)
        assert result.status == DataQualityStatus.INVALID

    def test_negative_price_is_invalid(self):
        result = validate_ohlc_bar(open_=-1, high=105, low=98, close=100, volume=1000)
        assert result.status == DataQualityStatus.INVALID

    def test_zero_price_is_invalid(self):
        result = validate_ohlc_bar(open_=0, high=105, low=98, close=100, volume=1000)
        assert result.status == DataQualityStatus.INVALID

    def test_negative_volume_is_invalid(self):
        result = validate_ohlc_bar(open_=100, high=105, low=98, close=103, volume=-5)
        assert result.status == DataQualityStatus.INVALID

    def test_missing_price_is_invalid(self):
        result = validate_ohlc_bar(open_=None, high=105, low=98, close=103, volume=1000)
        assert result.status == DataQualityStatus.INVALID

    def test_nan_price_is_invalid(self):
        result = validate_ohlc_bar(open_=float("nan"), high=105, low=98, close=103, volume=1000)
        assert result.status == DataQualityStatus.INVALID

    def test_large_move_is_suspect_not_invalid(self):
        """
        A real stock split or circuit-limit day can legitimately move a
        price 40%+ in a session — this must NOT be rejected outright
        (§3: "do not classify unusual market movements as fraud merely
        because they are unusual"), only flagged.
        """
        result = validate_ohlc_bar(open_=100, high=155, low=95, close=150, volume=5000)
        assert result.status == DataQualityStatus.SUSPECT
        assert result.is_usable

    def test_zero_volume_is_suspect_not_invalid(self):
        result = validate_ohlc_bar(open_=100, high=102, low=99, close=101, volume=0)
        assert result.status == DataQualityStatus.SUSPECT
        assert result.is_usable

    def test_ohlc_equal_is_valid(self):
        """A bar with no intraday movement at all (open=high=low=close) is arithmetically consistent, not invalid."""
        result = validate_ohlc_bar(open_=100, high=100, low=100, close=100, volume=500)
        assert result.status == DataQualityStatus.VALID
