"""Tests for SolarOnly fallback mode tracking and auto-recovery.

Covers:
- _enter_fallback_mode() / _exit_fallback_mode() lifecycle
- get_fallback_state() snapshot
- reset_fallback_state() clears all fields
- /health endpoint includes fallback_mode payload
- /stats endpoint includes fallback_mode payload
- /health/reset clears fallback state
- Config env var parsing for PW_TEDAPI_RECOVERY / PW_TEDAPI_PROBE_INTERVAL
"""
import time
import pytest
from unittest.mock import patch

from app.core.gateway_manager import GatewayManager


class TestFallbackModeLifecycle:
    """enter/exit fallback mode contract tests."""

    @pytest.fixture
    def gm(self):
        """Fresh GatewayManager with a TEDAPI gateway registered."""
        from app.models.gateway import Gateway
        m = GatewayManager()
        m._fallback_state["gw1"] = m._new_fallback_state()
        m._tedapi_probe_failures["gw1"] = 0
        return m

    def test_enter_sets_state(self, gm):
        """_enter_fallback_mode sets is_fallback_mode, records fallback_since."""
        before = time.time()
        gm._enter_fallback_mode("gw1", "test reason")

        state = gm._fallback_state["gw1"]
        assert state["is_fallback_mode"] is True
        assert state["fallback_since"] is not None
        assert state["fallback_since"] >= before
        assert state["recovery_attempts"] == 0
        assert state["last_recovery_attempt"] is None

    def test_enter_is_idempotent(self, gm):
        """Double enter is a no-op — fallback_since is not reset."""
        gm._enter_fallback_mode("gw1", "first")
        first_since = gm._fallback_state["gw1"]["fallback_since"]

        # Second enter must not overwrite fallback_since
        gm._enter_fallback_mode("gw1", "second")

        assert gm._fallback_state["gw1"]["fallback_since"] == first_since

    def test_exit_clears_all_fields(self, gm):
        """_exit_fallback_mode clears all state."""
        gm._enter_fallback_mode("gw1", "test")
        gm._fallback_state["gw1"]["recovery_attempts"] = 5
        gm._fallback_state["gw1"]["last_recovery_attempt"] = time.time()

        gm._exit_fallback_mode("gw1")

        state = gm._fallback_state["gw1"]
        assert state["is_fallback_mode"] is False
        assert state["fallback_since"] is None
        assert state["recovery_attempts"] == 0
        assert state["last_recovery_attempt"] is None

    def test_exit_is_idempotent(self, gm):
        """Double exit is a no-op — no error when called outside fallback."""
        gm._exit_fallback_mode("gw1")
        assert gm._fallback_state["gw1"]["is_fallback_mode"] is False

    def test_exit_unknown_gateway_is_safe(self, gm):
        """Exit on an untracked gateway should not raise."""
        gm._exit_fallback_mode("nonexistent")

    def test_full_lifecycle(self, gm):
        """Full enter → accumulate attempts → exit cycle resets everything."""
        gm._enter_fallback_mode("gw1", "probe timeout")
        gm._fallback_state["gw1"]["recovery_attempts"] = 3
        gm._fallback_state["gw1"]["last_recovery_attempt"] = time.time()

        gm._exit_fallback_mode("gw1")

        state = gm._fallback_state["gw1"]
        assert state["is_fallback_mode"] is False
        assert state["recovery_attempts"] == 0
        assert state["last_recovery_attempt"] is None


class TestGetFallbackState:
    """get_fallback_state() snapshot tests."""

    @pytest.fixture
    def gm(self):
        m = GatewayManager()
        m._fallback_state["gw1"] = m._new_fallback_state()
        return m

    def test_returns_none_for_untracked(self, gm):
        """get_fallback_state returns None for a gateway without fallback tracking."""
        assert gm.get_fallback_state("untracked") is None

    def test_healthy_state_snapshot(self, gm):
        """get_fallback_state returns correct snapshot when healthy."""
        snapshot = gm.get_fallback_state("gw1")
        assert snapshot["is_fallback_mode"] is False
        assert snapshot["fallback_since"] is None
        assert snapshot["fallback_duration_seconds"] is None
        assert snapshot["recovery_attempts"] == 0
        assert snapshot["recovery_enabled"] is True  # default

    def test_fallback_state_snapshot(self, gm):
        """get_fallback_state returns correct snapshot when in fallback."""
        gm._enter_fallback_mode("gw1", "test")
        gm._fallback_state["gw1"]["recovery_attempts"] = 2

        snapshot = gm.get_fallback_state("gw1")
        assert snapshot["is_fallback_mode"] is True
        assert snapshot["fallback_since"] is not None
        assert snapshot["fallback_duration_seconds"] is not None
        assert snapshot["fallback_duration_seconds"] >= 0
        assert snapshot["recovery_attempts"] == 2


class TestResetFallbackState:
    """reset_fallback_state() tests."""

    @pytest.fixture
    def gm(self):
        m = GatewayManager()
        m._fallback_state["gw1"] = m._new_fallback_state()
        m._fallback_state["gw2"] = m._new_fallback_state()
        m._tedapi_probe_failures["gw1"] = 5
        m._tedapi_probe_failures["gw2"] = 3
        return m

    def test_reset_single_gateway(self, gm):
        """reset_fallback_state(gateway_id) clears one gateway."""
        gm._enter_fallback_mode("gw1", "test")
        gm.reset_fallback_state("gw1")

        assert gm._fallback_state["gw1"]["is_fallback_mode"] is False
        assert gm._tedapi_probe_failures["gw1"] == 0
        # gw2 unaffected
        assert gm._tedapi_probe_failures["gw2"] == 3

    def test_reset_all_gateways(self, gm):
        """reset_fallback_state() with no args clears all."""
        gm._enter_fallback_mode("gw1", "test")
        gm._enter_fallback_mode("gw2", "test")

        gm.reset_fallback_state()

        assert gm._fallback_state["gw1"]["is_fallback_mode"] is False
        assert gm._fallback_state["gw2"]["is_fallback_mode"] is False
        assert gm._tedapi_probe_failures["gw1"] == 0
        assert gm._tedapi_probe_failures["gw2"] == 0


class TestHealthEndpointFallback:
    """/health endpoint includes fallback_mode payload."""

    def test_health_no_fallback(self, client):
        """/health works when no fallback state exists."""
        # Autouse fixture already clears singleton state — no gateways means
        # the health endpoint returns "no_gateways" without fallback data.
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_gateways"

    def test_health_with_fallback(self, client):
        """/health includes fallback_mode when a gateway is in fallback."""
        from app.core.gateway_manager import GatewayManager

        test_gm = GatewayManager()
        test_gm._fallback_state["default"] = test_gm._new_fallback_state()
        test_gm._enter_fallback_mode("default", "test probe failure")

        with patch("app.main.gateway_manager", test_gm), \
             patch("app.api.legacy.gateway_manager", test_gm):
            from app.models.gateway import Gateway
            test_gm.gateways["default"] = Gateway(
                id="default", name="Test", online=True
            )
            from app.models.gateway import GatewayStatus
            test_gm.cache["default"] = GatewayStatus(
                gateway=test_gm.gateways["default"], online=True
            )

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "fallback_mode" in data
            assert "default" in data["fallback_mode"]
            assert data["fallback_mode"]["default"]["is_fallback_mode"] is True


class TestStatsEndpointFallback:
    """/stats endpoint includes fallback_mode payload."""

    def test_stats_includes_fallback_when_tracked(self, client):
        """/stats includes fallback_mode when a gateway has fallback state."""
        from app.core.gateway_manager import GatewayManager

        test_gm = GatewayManager()
        test_gm._fallback_state["default"] = test_gm._new_fallback_state()

        with patch("app.api.legacy.gateway_manager", test_gm):
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "fallback_mode" in data
            assert "default" in data["fallback_mode"]
            assert data["fallback_mode"]["default"]["is_fallback_mode"] is False


class TestHealthResetEndpoint:
    """/health/reset clears fallback state."""

    def test_health_reset_requires_auth(self, client):
        """POST /health/reset returns 403 when control secret is not set."""
        response = client.post("/health/reset")
        assert response.status_code == 403

    def test_health_reset_returns_ok_with_token(self, client, monkeypatch):
        """POST /health/reset returns ok with a valid control token."""
        from app.config import settings
        monkeypatch.setattr(settings, "control_secret", "test-secret")
        response = client.post(
            "/health/reset",
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "fallback" in data["message"].lower()


class TestConfigSettings:
    """Config env var parsing for TEDAPI recovery settings."""

    def test_defaults(self):
        """PW_TEDAPI_RECOVERY defaults to True, PW_TEDAPI_PROBE_INTERVAL to 30."""
        from app.config import Settings
        import os

        # Temporarily clear env vars
        old_recovery = os.environ.pop("PW_TEDAPI_RECOVERY", None)
        old_interval = os.environ.pop("PW_TEDAPI_PROBE_INTERVAL", None)
        try:
            s = Settings()
            assert s.tedapi_recovery is True
            assert s.tedapi_probe_interval == 30
        finally:
            if old_recovery is not None:
                os.environ["PW_TEDAPI_RECOVERY"] = old_recovery
            if old_interval is not None:
                os.environ["PW_TEDAPI_PROBE_INTERVAL"] = old_interval

    def test_disable_recovery(self):
        """PW_TEDAPI_RECOVERY=no disables recovery."""
        from app.config import Settings
        import os

        old = os.environ.get("PW_TEDAPI_RECOVERY")
        os.environ["PW_TEDAPI_RECOVERY"] = "no"
        try:
            s = Settings()
            assert s.tedapi_recovery is False
        finally:
            if old is None:
                os.environ.pop("PW_TEDAPI_RECOVERY", None)
            else:
                os.environ["PW_TEDAPI_RECOVERY"] = old

    def test_custom_probe_interval(self):
        """PW_TEDAPI_PROBE_INTERVAL can be customized."""
        from app.config import Settings
        import os

        old = os.environ.get("PW_TEDAPI_PROBE_INTERVAL")
        os.environ["PW_TEDAPI_PROBE_INTERVAL"] = "10"
        try:
            s = Settings()
            assert s.tedapi_probe_interval == 10
        finally:
            if old is None:
                os.environ.pop("PW_TEDAPI_PROBE_INTERVAL", None)
            else:
                os.environ["PW_TEDAPI_PROBE_INTERVAL"] = old
