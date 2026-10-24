"""Offline tests for the standalone Runpod watchdog control logic."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from urllib.request import Request

from tools.runpod_watchdog import (
    API_BASE,
    Outcome,
    PodIdentity,
    Policy,
    RunpodClient,
    _NoRedirectHandler,
    atomic_receipt,
    build_parser,
    check_once,
    parse_timestamp,
    policy_from_args,
)

NOW = 1_800_000_000.0
CREATED_AT = "2027-01-15T08:00:00Z"


def pod(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "id": "sample-pod-id",
        "name": "sample-pilot",
        "gpu": {"id": "NVIDIA A40", "count": 1},
        "dataCenterId": "US-XX-1",
        "createdAt": CREATED_AT,
        "cost": 0.5,
        "status": "RUNNING",
    }
    result.update(changes)
    return result


IDENTITY = PodIdentity.from_dict(
    {
        "id": "sample-pod-id",
        "name": "sample-pilot",
        "gpuId": "NVIDIA A40",
        "gpuCount": 1,
        "dataCenterId": "US-XX-1",
        "createdAt": CREATED_AT,
    }
)
POLICY = Policy(
    5400,
    2.0,
    600,
    0.5,
    delete_verify_attempts=2,
    delete_verify_delay_seconds=0,
)


class FakeTransport:
    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self.responses = responses.copy()
        self.calls: list[tuple[str, str, dict[str, str], float]] = []

    def request(
        self, method: str, url: str, headers: dict[str, str], timeout: float
    ) -> tuple[int, bytes]:
        self.calls.append((method, url, headers, timeout))
        return self.responses.pop(0)


def json_response(status: int, value: object) -> tuple[int, bytes]:
    return status, json.dumps(value).encode()


def heartbeat(
    path: Path,
    *,
    activity: float = NOW - 20,
    updated: float = NOW - 2,
    status: str = "running",
    mac_receipt_verified: bool = False,
    github_receipt_verified: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "updated_at_unix": updated,
                "last_gpu_activity_unix": activity,
                "phase": "train",
                "step": 4,
                "status": status,
                "mac_receipt_verified": mac_receipt_verified,
                "github_receipt_verified": github_receipt_verified,
            }
        ),
        encoding="utf-8",
    )


class RunpodWatchdogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.heartbeat_path = Path(self.temp.name) / "heartbeat.json"
        heartbeat(self.heartbeat_path)

    def client(self, responses: list[tuple[int, bytes]]) -> tuple[RunpodClient, FakeTransport]:
        transport = FakeTransport(responses)
        return RunpodClient("secret-placeholder", transport), transport

    def test_wall_deadline_requires_delete_204_and_get_404(self) -> None:
        client, transport = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            POLICY,
            self.heartbeat_path,
            now=lambda: NOW + 5400,
            sleep=lambda _: None,
        )
        self.assertEqual(result.state, "terminated")
        self.assertEqual(result.reason, "wall_limit_delete_204_get_404")
        self.assertEqual([call[0] for call in transport.calls], ["GET", "DELETE", "GET"])
        self.assertTrue(
            all(call[1] == API_BASE + "/pods/sample-pod-id" for call in transport.calls)
        )

    def test_pre_staging_cost_counts_toward_the_stop_cost(self) -> None:
        client, transport = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        policy = replace(POLICY, max_cost_usd=1.0, starting_cost_usd=0.99)
        result = check_once(
            client,
            IDENTITY,
            policy,
            self.heartbeat_path,
            now=lambda: NOW + 100,
            sleep=lambda _: None,
        )
        self.assertEqual(result.state, "terminated")
        self.assertEqual(result.reason, "cost_limit_delete_204_get_404")
        with self.assertRaises(ValueError):
            replace(policy, starting_cost_usd=1.0)
        self.assertTrue(
            all(call[2]["Authorization"] == "Bearer secret-placeholder" for call in transport.calls)
        )
        self.assertTrue(all(call[3] == POLICY.request_timeout_seconds for call in transport.calls))

    def test_identity_mismatch_never_deletes(self) -> None:
        different = pod(name="different-name")
        client, transport = self.client([json_response(200, different)])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 6000)
        self.assertEqual(result.state, "identity_mismatch")
        self.assertEqual(result.reason, "identity_mismatch_no_delete")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])

    def test_pod_id_rejects_path_traversal_before_transport(self) -> None:
        client, transport = self.client([])
        with self.assertRaises(ValueError):
            client.get_pod("../other-pod", 1)
        self.assertEqual(transport.calls, [])

    def test_gpu_identity_mismatch_never_deletes(self) -> None:
        different = pod(gpu={"id": "NVIDIA A100", "count": 1})
        client, transport = self.client([json_response(200, different)])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 6000)
        self.assertEqual(result.state, "identity_mismatch")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])

    def test_all_captured_identity_fields_are_required_before_delete(self) -> None:
        variants = [
            pod(id="other-pod-id"),
            pod(gpu={"id": "NVIDIA A40", "count": 2}),
            pod(dataCenterId="US-YY-1"),
            pod(createdAt="2027-01-15T08:01:00Z"),
        ]
        for different in variants:
            with self.subTest(pod=different):
                client, transport = self.client([json_response(200, different)])
                result = check_once(
                    client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 6000
                )
                self.assertEqual(result.state, "identity_mismatch")
                self.assertEqual([call[0] for call in transport.calls], ["GET"])

    def test_api_authorization_error_is_unknown_and_never_deletes(self) -> None:
        client, transport = self.client([(403, b'{"error":"denied"}')])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 6000)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.reason, "pod_get_http_403")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])

    def test_cost_guard_terminates_after_matching_identity(self) -> None:
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            Policy(5400, 0.01, 600, 0.5, delete_verify_attempts=1),
            self.heartbeat_path,
            now=lambda: NOW + 600,
            sleep=lambda _: None,
        )
        self.assertEqual(result.reason, "cost_limit_delete_204_get_404")

    def test_stale_gpu_heartbeat_terminates(self) -> None:
        heartbeat(self.heartbeat_path, activity=NOW - 600, updated=NOW - 1)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_malformed_heartbeat_fails_closed_after_identity_check(self) -> None:
        self.heartbeat_path.write_text("{broken", encoding="utf-8")
        client, transport = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.reason, "heartbeat_malformed_delete_204_get_404")
        self.assertEqual([call[0] for call in transport.calls], ["GET", "DELETE", "GET"])

    def test_missing_first_heartbeat_uses_pod_creation_as_idle_baseline(self) -> None:
        self.heartbeat_path.unlink()
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            POLICY,
            self.heartbeat_path,
            now=lambda: NOW + 2000,
            sleep=lambda _: None,
        )
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_stale_worker_heartbeat_fails_closed(self) -> None:
        heartbeat(self.heartbeat_path, activity=NOW - 901, updated=NOW - 900)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.reason, "heartbeat_stale_delete_204_get_404")

    def test_terminal_worker_statuses_wait_for_transfer_with_idle_limit(self) -> None:
        for status in ("awaiting_transfer", "completed", "time_limited"):
            with self.subTest(status=status):
                heartbeat(self.heartbeat_path, activity=NOW, updated=NOW + 10, status=status)
                client, transport = self.client([json_response(200, pod())])
                result = check_once(
                    client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 10
                )
                self.assertEqual(result.state, "watching")
                self.assertEqual([call[0] for call in transport.calls], ["GET"])

    def test_awaiting_transfer_terminates_after_idle_window(self) -> None:
        heartbeat(
            self.heartbeat_path,
            activity=NOW,
            updated=NOW + 590,
            status="awaiting_transfer",
        )
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 600)
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_shutdown_ready_holds_until_both_transfer_receipts_are_verified(self) -> None:
        heartbeat(
            self.heartbeat_path,
            activity=NOW - 590,
            updated=NOW - 1,
            status="shutdown_ready",
            mac_receipt_verified=True,
            github_receipt_verified=False,
        )
        client, transport = self.client([json_response(200, pod())])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 5)
        self.assertEqual(result.state, "watching")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])

    def test_shutdown_ready_deletes_after_both_transfer_receipts_are_verified(self) -> None:
        heartbeat(
            self.heartbeat_path,
            activity=NOW,
            updated=NOW + 2,
            status="shutdown_ready",
            mac_receipt_verified=True,
            github_receipt_verified=True,
        )
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 3)
        self.assertEqual(result.reason, "shutdown_ready_verified_receipts_delete_204_get_404")

    def test_unverified_shutdown_ready_still_deletes_after_idle_limit(self) -> None:
        heartbeat(
            self.heartbeat_path,
            activity=NOW - 601,
            updated=NOW - 1,
            status="shutdown_ready",
            mac_receipt_verified=True,
            github_receipt_verified=False,
        )
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_malformed_shutdown_ready_heartbeat_cannot_authorize_receipt_delete(self) -> None:
        heartbeat(
            self.heartbeat_path,
            activity=NOW + 10,
            updated=NOW + 2,
            status="shutdown_ready",
            mac_receipt_verified=True,
            github_receipt_verified=True,
        )
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 20)
        self.assertEqual(result.reason, "heartbeat_malformed_delete_204_get_404")

    def test_stopped_and_failed_worker_wait_for_receipt_or_idle_limit(self) -> None:
        for status in ("stopped", "failed"):
            with self.subTest(status=status):
                heartbeat(self.heartbeat_path, activity=NOW, updated=NOW, status=status)
                client, transport = self.client([json_response(200, pod())])
                result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
                self.assertEqual(result.state, "watching")
                self.assertEqual([call[0] for call in transport.calls], ["GET"])

                client, transport = self.client([json_response(200, pod()), (204, b""), (404, b"")])
                result = check_once(
                    client,
                    IDENTITY,
                    POLICY,
                    self.heartbeat_path,
                    now=lambda: NOW + POLICY.max_idle_seconds,
                    sleep=lambda _: None,
                )
                self.assertIn(
                    result.reason,
                    {
                        "idle_gpu_limit_delete_204_get_404",
                        "heartbeat_stale_delete_204_get_404",
                    },
                )
                self.assertEqual([call[0] for call in transport.calls], ["GET", "DELETE", "GET"])

    def test_failed_worker_waits_for_receipt_or_wall_deadline(self) -> None:
        near_wall = parse_timestamp(CREATED_AT) + POLICY.max_wall_seconds - 1
        heartbeat(
            self.heartbeat_path,
            activity=near_wall,
            updated=near_wall,
            status="failed",
        )
        client, transport = self.client([json_response(200, pod())])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: near_wall)
        self.assertEqual(result.state, "watching")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])

        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            POLICY,
            self.heartbeat_path,
            now=lambda: parse_timestamp(CREATED_AT) + POLICY.max_wall_seconds,
            sleep=lambda _: None,
        )
        self.assertEqual(result.reason, "wall_limit_delete_204_get_404")

    def test_quoted_total_hourly_floors_cost_above_api_cost(self) -> None:
        client, _ = self.client([json_response(200, pod(cost=0.01)), (204, b""), (404, b"")])
        policy = Policy(5400, 0.02, 600, 2.0, delete_verify_attempts=1)
        result = check_once(client, IDENTITY, policy, self.heartbeat_path, now=lambda: NOW + 60)
        self.assertEqual(result.reason, "cost_limit_delete_204_get_404")

    def test_billing_start_before_pod_created_is_used_for_elapsed_guards(self) -> None:
        created = parse_timestamp(CREATED_AT)
        policy = Policy(5400, 10.0, 600, 0.5, billing_started_at_unix=created - 1000)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client, IDENTITY, policy, self.heartbeat_path, now=lambda: created + 4500
        )
        self.assertEqual(result.reason, "wall_limit_delete_204_get_404")

    def test_billing_start_after_pod_creation_fails_closed(self) -> None:
        created = parse_timestamp(CREATED_AT)
        policy = Policy(5400, 10.0, 600, 0.5, billing_started_at_unix=created + 120)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, policy, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.reason, "billing_start_after_pod_creation_delete_204_get_404")
        # A Mac clock a few seconds ahead of the provider's is tolerated, not fatal.
        tolerated = Policy(5400, 10.0, 600, 0.5, billing_started_at_unix=created + 30)
        client, transport = self.client([json_response(200, pod())])
        result = check_once(client, IDENTITY, tolerated, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.state, "watching")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])

    def test_created_at_rendering_differences_still_match_identity(self) -> None:
        rendered = pod(createdAt="2027-01-15T08:00:00.000+00:00")
        client, _ = self.client([json_response(200, rendered)])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.state, "watching")
        shifted = pod(createdAt="2027-01-15T08:00:02Z")
        client, _ = self.client([json_response(200, shifted)])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.state, "identity_mismatch")

    def test_slightly_fast_pod_clock_is_not_a_malformed_heartbeat(self) -> None:
        heartbeat(self.heartbeat_path, activity=NOW + 5, updated=NOW + 10)
        client, transport = self.client([json_response(200, pod())])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.state, "watching")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])
        heartbeat(self.heartbeat_path, activity=NOW, updated=NOW + 120)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.reason, "heartbeat_malformed_delete_204_get_404")

    def test_redirect_handler_refuses_forwarding_authorization(self) -> None:
        handler = _NoRedirectHandler()
        request = Request(API_BASE, headers={"Authorization": "Bearer test"})
        result = handler.redirect_request(
            request, None, 302, "Found", {}, "https://attacker.example/collect"
        )
        self.assertIsNone(result)

    def test_future_gpu_activity_after_heartbeat_update_is_malformed(self) -> None:
        heartbeat(self.heartbeat_path, activity=NOW + 10, updated=NOW + 2)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW + 20)
        self.assertEqual(result.reason, "heartbeat_malformed_delete_204_get_404")

    def test_delete_error_is_unknown_not_success(self) -> None:
        client, _ = self.client([json_response(200, pod()), (500, b"failure")])
        result = check_once(
            client,
            IDENTITY,
            POLICY,
            self.heartbeat_path,
            now=lambda: NOW + 5400,
        )
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.reason, "wall_limit_delete_http_500")

    def test_delete_204_without_absence_is_unknown(self) -> None:
        client, _ = self.client(
            [
                json_response(200, pod()),
                (204, b""),
                json_response(200, pod()),
                json_response(200, pod()),
            ]
        )
        result = check_once(
            client,
            IDENTITY,
            POLICY,
            self.heartbeat_path,
            now=lambda: NOW + 5400,
            sleep=lambda _: None,
        )
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.reason, "wall_limit_delete_still_present")

    def test_receipt_is_atomic_json_and_contains_no_secret(self) -> None:
        receipt = Path(self.temp.name) / "receipts" / "watchdog.json"
        atomic_receipt(receipt, Outcome("watching", "within_limits", IDENTITY.pod_id, NOW))
        result = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertEqual(result["state"], "watching")
        self.assertNotIn("secret-placeholder", receipt.read_text(encoding="utf-8"))
        self.assertEqual(list(receipt.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()


class ObservationAndTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.heartbeat_path = Path(self.temp.name) / "heartbeat.json"
        heartbeat(self.heartbeat_path)

    def client(self, responses: list[tuple[int, bytes]]) -> tuple[RunpodClient, FakeTransport]:
        transport = FakeTransport(responses)
        return RunpodClient("secret-placeholder", transport), transport

    def test_first_poll_404_is_unknown_not_absent(self) -> None:
        # A wrong base URL, Pod ID or key scope must not read as a completed shutdown.
        client, transport = self.client([(404, b"")])
        result = check_once(client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.reason, "pod_get_404_before_identity_observed")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])
        client, _ = self.client([(404, b"")])
        observed = check_once(
            client, IDENTITY, POLICY, self.heartbeat_path, now=lambda: NOW, observed=True
        )
        self.assertEqual((observed.state, observed.reason), ("absent", "pod_get_404"))

    def test_observed_state_round_trips_through_the_state_file(self) -> None:
        from tools.runpod_watchdog import load_observed, save_observed

        path = Path(self.temp.name) / "receipts" / "watchdog.json.observed.json"
        self.assertFalse(load_observed(path, IDENTITY))
        save_observed(path, IDENTITY, NOW)
        self.assertTrue(load_observed(path, IDENTITY))
        other = PodIdentity.from_dict(
            {
                "id": "other-pod",
                "name": "sample-pilot",
                "gpuId": "NVIDIA A40",
                "gpuCount": 1,
                "dataCenterId": "US-XX-1",
                "createdAt": CREATED_AT,
            }
        )
        self.assertFalse(load_observed(path, other))

    def _transfer_heartbeat(
        self, *, progress: float | None, status: str = "awaiting_transfer"
    ) -> None:
        value = {
            "schema_version": 1,
            "updated_at_unix": NOW + 900,
            "last_gpu_activity_unix": NOW,
            "phase": "finalizing",
            "step": 4,
            "status": status,
        }
        if progress is not None:
            value["transfer_progress_unix"] = progress
        self.heartbeat_path.write_text(json.dumps(value), encoding="utf-8")

    def test_live_transfer_progress_extends_idle_within_transfer_window(self) -> None:
        policy = replace(POLICY, max_transfer_seconds=1500)
        self._transfer_heartbeat(progress=NOW + 890)
        client, transport = self.client([json_response(200, pod())])
        result = check_once(client, IDENTITY, policy, self.heartbeat_path, now=lambda: NOW + 900)
        self.assertEqual(result.state, "watching")
        self.assertEqual([call[0] for call in transport.calls], ["GET"])

    def test_stale_transfer_progress_or_no_window_still_terminates(self) -> None:
        self._transfer_heartbeat(progress=NOW + 200)  # Progress older than the idle limit.
        policy = replace(POLICY, max_transfer_seconds=1500)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            policy,
            self.heartbeat_path,
            now=lambda: NOW + 900,
            sleep=lambda _: None,
        )
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")
        self._transfer_heartbeat(progress=NOW + 890)  # Fresh progress but no window configured.
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            POLICY,
            self.heartbeat_path,
            now=lambda: NOW + 900,
            sleep=lambda _: None,
        )
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_transfer_window_itself_is_bounded(self) -> None:
        policy = replace(POLICY, max_transfer_seconds=800)
        self._transfer_heartbeat(progress=NOW + 890)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            policy,
            self.heartbeat_path,
            now=lambda: NOW + 900,
            sleep=lambda _: None,
        )
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_running_status_never_uses_the_transfer_window(self) -> None:
        policy = replace(POLICY, max_transfer_seconds=1500)
        self._transfer_heartbeat(progress=NOW + 890, status="running")
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            policy,
            self.heartbeat_path,
            now=lambda: NOW + 900,
            sleep=lambda _: None,
        )
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_future_transfer_progress_is_malformed(self) -> None:
        policy = replace(POLICY, max_transfer_seconds=1500)
        self._transfer_heartbeat(progress=NOW + 950)
        client, _ = self.client([json_response(200, pod()), (204, b""), (404, b"")])
        result = check_once(
            client,
            IDENTITY,
            policy,
            self.heartbeat_path,
            now=lambda: NOW + 900,
            sleep=lambda _: None,
        )
        self.assertEqual(result.reason, "heartbeat_malformed_delete_204_get_404")

    def test_transfer_window_policy_is_bounded(self) -> None:
        with self.assertRaises(ValueError):
            replace(POLICY, max_transfer_seconds=1801)
        with self.assertRaises(ValueError):
            replace(POLICY, max_transfer_seconds=-1)
        with self.assertRaises(ValueError):
            RunpodClient("k", FakeTransport([])).request_path("GET", "pods/../x", 1)


class IdleBaselineTests(unittest.TestCase):
    """Before the first heartbeat, an explicit idle baseline postpones only the idle limit."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.heartbeat_path = Path(self.temp.name) / "heartbeat.json"  # No heartbeat yet.

    def check(self, policy: Policy, now: float, responses: list[tuple[int, bytes]]) -> Outcome:
        client = RunpodClient("secret-placeholder", FakeTransport(responses))
        return check_once(
            client, IDENTITY, policy, self.heartbeat_path, now=lambda: now, sleep=lambda _: None
        )

    def deletes(self) -> list[tuple[int, bytes]]:
        return [json_response(200, pod()), (204, b""), (404, b"")]

    def test_baseline_postpones_the_first_heartbeat_idle_limit(self) -> None:
        with_baseline = replace(POLICY, idle_baseline_unix=NOW + 500)
        result = self.check(with_baseline, NOW + 700, [json_response(200, pod())])
        self.assertEqual((result.state, result.reason), ("watching", "within_limits"))
        # The same moment is 700 s after creation, so without a baseline it is idle.
        result = self.check(POLICY, NOW + 700, self.deletes())
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_baseline_still_expires_and_never_precedes_creation(self) -> None:
        result = self.check(
            replace(POLICY, idle_baseline_unix=NOW + 500), NOW + 1100, self.deletes()
        )
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")
        early = replace(POLICY, idle_baseline_unix=NOW - 5000)  # Clamped to Pod creation.
        result = self.check(early, NOW + 700, self.deletes())
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_future_baseline_fails_closed(self) -> None:
        result = self.check(
            replace(POLICY, idle_baseline_unix=NOW + 900), NOW + 100, self.deletes()
        )
        self.assertEqual(result.reason, "idle_baseline_invalid_delete_204_get_404")

    def test_baseline_never_extends_the_wall_limit(self) -> None:
        late = replace(POLICY, idle_baseline_unix=NOW + 5000)
        result = self.check(late, NOW + 5400, self.deletes())
        self.assertEqual(result.reason, "wall_limit_delete_204_get_404")

    def test_baseline_is_ignored_once_a_heartbeat_exists(self) -> None:
        heartbeat(self.heartbeat_path, activity=NOW + 10, updated=NOW + 690)
        policy = replace(POLICY, idle_baseline_unix=NOW + 500)
        result = self.check(policy, NOW + 700, self.deletes())
        self.assertEqual(result.reason, "idle_gpu_limit_delete_204_get_404")

    def test_policy_validates_the_baseline(self) -> None:
        for bad in (float("nan"), float("inf"), 0.0, -1.0, True, "1"):
            with self.assertRaises(ValueError):
                replace(POLICY, idle_baseline_unix=bad)  # type: ignore[arg-type]

    def test_cli_accepts_the_baseline(self) -> None:
        args = build_parser().parse_args(
            [
                "--identity",
                "i.json",
                "--heartbeat",
                "h.json",
                "--receipt",
                "r.json",
                "--max-wall-seconds",
                "5400",
                "--max-cost-usd",
                "1.08",
                "--quoted-total-hourly",
                "0.6",
                "--idle-baseline-unix",
                str(NOW + 30),
            ]
        )
        self.assertEqual(policy_from_args(args).idle_baseline_unix, NOW + 30)
