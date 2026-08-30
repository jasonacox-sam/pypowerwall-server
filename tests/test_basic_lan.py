"""Tests for Powerwall 3 Basic LAN mode (PW_HOST + PW_PASSWORD).

Covers discussion #79: PW3 units expose a limited local API over their wired
LAN interface (vendor subnet). PW_HOST + PW_PASSWORD must be accepted as a
valid local configuration, the connectivity probe must not rely on
/api/status (404 in this mode), and the poller must skip the endpoints this
mode does not serve.
"""
import pytest
from unittest.mock import Mock

from app.core.gateway_manager import gateway_manager
from app.core.scaling import raw_to_tesla_battery_percent
from app.config import GatewayConfig
from app.models.gateway import Gateway, GatewayStatus


def _make_basic_lan_pw_mock() -> Mock:
    """Mock pypowerwall connection shaped like PW3 Basic LAN behavior.

    - /api/status is NOT served (is_connected() would be a false negative)
    - /api/meters/aggregates, /api/system_status/soe and
      /api/system_status/grid_status ARE served
    - no TEDAPI client attached
    """
    mock = Mock()
    mock.poll.side_effect = lambda api, **kw: {
        "/api/meters/aggregates": {
            "site": {"instant_power": -6648, "instant_reactive_power": 0},
            "solar": {"instant_power": 1497, "instant_reactive_power": 0},
            "battery": {"instant_power": -881, "instant_reactive_power": 0},
            "load": {"instant_power": 621, "instant_reactive_power": 0},
        },
        "/api/system_status/grid_status": {"grid_status": "SystemGridConnected"},
    }.get(api)
    mock.level.return_value = 90.27777777777779
    mock.grid_status.return_value = "UP"
    mock.is_connected.return_value = False  # /api/status is 404 in Basic LAN
    mock.tedapi = None
    return mock


@pytest.mark.asyncio
async def test_initialize_accepts_host_plus_password(monkeypatch):
    """host + password (Basic LAN) registers as a valid local gateway."""
    import pypowerwall

    mock_pw = _make_basic_lan_pw_mock()
    monkeypatch.setattr(pypowerwall, "Powerwall", lambda **kw: mock_pw)

    configs = [
        GatewayConfig(id="pw3", name="PW3", host="10.42.1.44", password="12345")
    ]

    await gateway_manager.initialize(configs, poll_interval=5)

    assert "pw3" in gateway_manager.gateways, (
        "host+password config was rejected - Basic LAN must be a valid local mode"
    )
    assert gateway_manager.gateways["pw3"].basic_lan is True

    await gateway_manager.shutdown()


@pytest.mark.asyncio
async def test_initialize_accepts_legacy_pw_password_env_setting(monkeypatch):
    """host + settings.pw_password (PW_PASSWORD env var) is also accepted."""
    from app.config import settings

    import pypowerwall

    mock_pw = _make_basic_lan_pw_mock()
    monkeypatch.setattr(pypowerwall, "Powerwall", lambda **kw: mock_pw)
    monkeypatch.setattr(settings, "pw_password", "12345")

    configs = [
        GatewayConfig(id="pw3-env", name="PW3", host="10.42.1.44"),
    ]

    await gateway_manager.initialize(configs, poll_interval=5)

    assert "pw3-env" in gateway_manager.gateways
    assert gateway_manager.gateways["pw3-env"].basic_lan is True

    await gateway_manager.shutdown()


@pytest.mark.asyncio
async def test_initialize_still_rejects_host_only(monkeypatch):
    """host with no credentials of any kind is still invalid."""
    configs = [GatewayConfig(id="bad", name="Bad", host="10.0.0.5")]

    await gateway_manager.initialize(configs, poll_interval=5)

    assert "bad" not in gateway_manager.gateways

    await gateway_manager.shutdown()


@pytest.mark.asyncio
async def test_basic_lan_probe_falls_back_to_grid_status(
    mock_gateway_manager, mock_pypowerwall
):
    """is_connected() false negative must not reject a reachable Basic LAN gateway.

    PW3 Basic LAN returns 404 for /api/status, so pypowerwall's is_connected()
    reports False even though the mode's endpoints answer. The poller must
    fall back to pw.grid_status() for the connectivity probe.
    """
    mock_pypowerwall.is_connected.return_value = False
    mock_pypowerwall.grid_status.return_value = "UP"

    gw = Gateway(id="pw3-probe", name="PW3", host="10.42.1.44", basic_lan=True)
    config = GatewayConfig(
        id="pw3-probe", name="PW3", host="10.42.1.44", password="12345"
    )

    gateway_manager.gateways["pw3-probe"] = gw
    gateway_manager._pending_configs["pw3-probe"] = config
    gateway_manager.cache["pw3-probe"] = GatewayStatus(gateway=gw, online=False)
    gateway_manager._consecutive_failures["pw3-probe"] = 0
    gateway_manager._next_poll_time["pw3-probe"] = 0

    await gateway_manager._poll_gateway("pw3-probe")

    # Connection established despite is_connected() == False
    assert "pw3-probe" in gateway_manager.connections
    mock_pypowerwall.grid_status.assert_called()
    assert gateway_manager.cache["pw3-probe"].online is True


@pytest.mark.asyncio
async def test_basic_lan_probe_fails_when_grid_status_unavailable(
    mock_gateway_manager, mock_pypowerwall
):
    """A genuinely unreachable Basic LAN gateway must still fail cleanly."""
    mock_pypowerwall.is_connected.return_value = False
    mock_pypowerwall.grid_status.return_value = None

    gw = Gateway(id="pw3-dead", name="PW3", host="10.42.1.44", basic_lan=True)
    config = GatewayConfig(
        id="pw3-dead", name="PW3", host="10.42.1.44", password="12345"
    )

    gateway_manager.gateways["pw3-dead"] = gw
    gateway_manager._pending_configs["pw3-dead"] = config
    gateway_manager.cache["pw3-dead"] = GatewayStatus(gateway=gw, online=False)
    gateway_manager._consecutive_failures["pw3-dead"] = 0
    gateway_manager._next_poll_time["pw3-dead"] = 0

    await gateway_manager._poll_gateway("pw3-dead")

    assert "pw3-dead" not in gateway_manager.connections
    assert "pw3-dead" in gateway_manager._pending_configs  # retained for retry
    assert gateway_manager.cache["pw3-dead"].online is False


@pytest.mark.asyncio
async def test_basic_lan_skips_unavailable_endpoint_fetches(
    mock_gateway_manager, mock_pypowerwall
):
    """Basic LAN gateways only poll the endpoints the mode serves."""
    mock_pypowerwall.tedapi = None

    gw = Gateway(id="pw3-fetch", name="PW3", host="10.42.1.44", basic_lan=True)
    gateway_manager.gateways["pw3-fetch"] = gw
    gateway_manager.connections["pw3-fetch"] = mock_pypowerwall

    data = await gateway_manager._fetch_gateway_data("pw3-fetch", mock_pypowerwall)

    # Core data is collected
    assert data.aggregates
    assert data.soe_raw == 85.5
    assert data.soe == pytest.approx(raw_to_tesla_battery_percent(85.5))
    assert data.grid_status == "UP"

    # Endpoints Basic LAN does not serve are never requested
    mock_pypowerwall.vitals.assert_not_called()
    mock_pypowerwall.strings.assert_not_called()
    mock_pypowerwall.status.assert_not_called()
    mock_pypowerwall.version.assert_not_called()
    mock_pypowerwall.din.assert_not_called()
    mock_pypowerwall.uptime.assert_not_called()
    mock_pypowerwall.alerts.assert_not_called()
    mock_pypowerwall.temps.assert_not_called()
    mock_pypowerwall.site_name.assert_not_called()
    mock_pypowerwall.get_mode.assert_not_called()
    mock_pypowerwall.get_reserve.assert_not_called()
    mock_pypowerwall.system_status.assert_not_called()

    # Direct polls limited to the two endpoints this mode serves
    polled = {c.args[0] for c in mock_pypowerwall.poll.call_args_list}
    assert polled == {"/api/meters/aggregates", "/api/system_status/grid_status"}


@pytest.mark.asyncio
async def test_basic_lan_hybrid_reads_mode_from_cloud(
    mock_gateway_manager, mock_pypowerwall
):
    """Hybrid Basic LAN: mode/reserve refresh from the cloud control connection.

    Regression (nesys, PR #85): the local Basic LAN API has no operation
    mode/reserve endpoint, so the Console showed a stale mode from the last
    cache write instead of the real system state. With a cloud control
    connection present, mode/reserve must be read from the cloud each poll.
    """
    mock_pypowerwall.tedapi = None

    mock_cloud = Mock()
    mock_cloud.get_mode.return_value = "autonomous"
    mock_cloud.get_reserve.return_value = 12.0
    gateway_manager._cloud_control = mock_cloud

    gw = Gateway(id="pw3-hybrid", name="PW3", host="10.42.1.44", basic_lan=True)
    gateway_manager.gateways["pw3-hybrid"] = gw
    gateway_manager.connections["pw3-hybrid"] = mock_pypowerwall

    data = await gateway_manager._fetch_gateway_data(
        "pw3-hybrid", mock_pypowerwall
    )

    # Mode/reserve come from the cloud control connection
    mock_cloud.get_mode.assert_called_once()
    mock_cloud.get_reserve.assert_called_once()
    assert mock_cloud.get_reserve.call_args.kwargs.get("scale") is True
    assert data.mode == "autonomous"
    assert data.reserve == 12.0

    # The local connection is still never asked for mode/reserve
    mock_pypowerwall.get_mode.assert_not_called()
    mock_pypowerwall.get_reserve.assert_not_called()


@pytest.mark.asyncio
async def test_hybrid_poll_reads_do_not_flood_info_logs(
    mock_gateway_manager, mock_pypowerwall, caplog
):
    """Hybrid poll-loop reads (get_mode/get_reserve) must log at DEBUG, not INFO.

    Regression (nesys, PR #85): the cloud refresh added with hybrid mode
    made cloud_control() succeed twice per poll cycle (~5s), and its INFO
    "completed successfully" lines flooded the logs. Reads log at DEBUG;
    user-initiated writes keep the INFO line.
    """
    import logging

    mock_pypowerwall.tedapi = None

    mock_cloud = Mock()
    mock_cloud.get_mode.return_value = "autonomous"
    mock_cloud.get_reserve.return_value = 12.0
    mock_cloud.set_reserve.return_value = True
    gateway_manager._cloud_control = mock_cloud

    gw = Gateway(id="pw3-lognoise", name="PW3", host="10.42.1.44", basic_lan=True)
    gateway_manager.gateways["pw3-lognoise"] = gw
    gateway_manager.connections["pw3-lognoise"] = mock_pypowerwall

    with caplog.at_level(logging.INFO):
        await gateway_manager._fetch_gateway_data("pw3-lognoise", mock_pypowerwall)
        assert "cloud_control(get_mode) completed" not in caplog.text
        assert "cloud_control(get_reserve) completed" not in caplog.text

    # Writes are user-initiated and rare - they stay visible at INFO
    with caplog.at_level(logging.INFO):
        await gateway_manager.cloud_control("set_reserve", 20)
        assert "cloud_control(set_reserve) completed successfully" in caplog.text


@pytest.mark.asyncio
async def test_basic_lan_without_cloud_does_not_show_stale_mode(
    mock_gateway_manager, mock_pypowerwall
):
    """Plain Basic LAN (no cloud): a cached mode must not be re-served stale.

    There is no source of truth for mode in this mode, so the previous
    value should be dropped rather than shown forever.
    """
    mock_pypowerwall.tedapi = None

    gw = Gateway(id="pw3-stale", name="PW3", host="10.42.1.44", basic_lan=True)
    gateway_manager.gateways["pw3-stale"] = gw
    gateway_manager.connections["pw3-stale"] = mock_pypowerwall

    stale = Mock()
    stale.mode = "self_consumption"
    gateway_manager._last_successful_data["pw3-stale"] = stale

    data = await gateway_manager._fetch_gateway_data(
        "pw3-stale", mock_pypowerwall
    )

    assert data.mode is None


@pytest.mark.asyncio
async def test_tedapi_gateway_still_fetches_optional_data(
    mock_gateway_manager, mock_pypowerwall
):
    """Non-Basic-LAN gateways keep fetching optional data (no behavior change)."""
    gw = Gateway(id="tedapi-fetch", name="GW", host="192.168.91.1", gw_pwd="secret")
    gateway_manager.gateways["tedapi-fetch"] = gw
    gateway_manager.connections["tedapi-fetch"] = mock_pypowerwall

    data = await gateway_manager._fetch_gateway_data("tedapi-fetch", mock_pypowerwall)

    mock_pypowerwall.vitals.assert_called()
    mock_pypowerwall.status.assert_called()
    mock_pypowerwall.get_mode.assert_called()
    assert data.vitals is not None
    assert data.mode == "self_consumption"


@pytest.mark.asyncio
async def test_stats_reports_basic_lan_mode(monkeypatch):
    """/stats must expose basiclan=True (and tedapi=False) for Basic LAN gateways.

    Regression: the Console connect-mode card showed "TEDAPI" for Basic LAN
    because /stats set tedapi=True for any gateway with a host (discussion #79).
    """
    import pypowerwall
    from app.main import app
    from httpx import ASGITransport, AsyncClient

    mock_pw = _make_basic_lan_pw_mock()
    monkeypatch.setattr(pypowerwall, "Powerwall", lambda **kw: mock_pw)

    configs = [
        GatewayConfig(id="pw3", name="PW3", host="10.42.1.44", password="12345")
    ]
    await gateway_manager.initialize(configs, poll_interval=5)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["basiclan"] is True
        assert data["tedapi"] is False

    await gateway_manager.shutdown()
