"""Offline checks for limits, cancellation, and stable source/result mapping."""

import ssl
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from threading import Event
from time import perf_counter
from unittest.mock import patch

import httpx

from modules.translation.batching import (
    RequestBudget,
    ServiceCooldown,
    TextPart,
    TranslationCancelled,
    _ssl_context,
    batch_results,
    join_parts,
    make_client,
    map_bounded,
    pack_parts,
    retry_delay,
    text_parts,
    validate_timeout,
)
from modules.translation.text_chunks import split_text


class BatchingTests(unittest.TestCase):
    def test_clients_share_verified_ssl_context_and_honor_custom_ca(self):
        _ssl_context.cache_clear()
        with patch.dict("os.environ", {"SSL_CERT_FILE": "", "SSL_CERT_DIR": ""}):
            first = make_client(5)
            second = make_client(5)
            self.addCleanup(first.close)
            self.addCleanup(second.close)
            context = _ssl_context(None, None)
            self.assertEqual(_ssl_context.cache_info().misses, 1)
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
        with patch(
            "modules.translation.batching._ssl_context", return_value=context
        ) as factory:
            with patch.dict(
                "os.environ",
                {"SSL_CERT_FILE": "custom.pem", "SSL_CERT_DIR": "custom-dir"},
            ):
                with make_client(5):
                    factory.assert_called_with("custom.pem", None)
                with make_client(5, trust_env=False):
                    factory.assert_called_with(None, None)
                with patch.dict("os.environ", {"SSL_CERT_FILE": ""}), make_client(5):
                    factory.assert_called_with(None, "custom-dir")

    def test_limits_and_unicode_preserve_all_words(self):
        source = "Привет мир. " * 100 + "x" * 30
        chunks = split_text(source, 17)
        self.assertTrue(all(0 < len(chunk) <= 17 for chunk in chunks))
        self.assertEqual("".join(source.split()), "".join("".join(chunks).split()))
        self.assertEqual(split_text(" \n\t" * 20, 3), [])

    def test_invalid_limits(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                split_text("text", value)
        with self.assertRaises(ValueError):
            pack_parts([TextPart(0, "too long")], 2)
        with ThreadPoolExecutor(1) as pool, self.assertRaises(ValueError):
            map_bounded(pool, str, [1], RequestBudget(), 0)

    def test_mapping_across_character_and_item_limits(self):
        texts = ["a" * 12, "b" * 4, "c" * 5, "d"]
        parts = text_parts(texts, 7)
        batches = pack_parts(parts, 7, 2)
        self.assertTrue(
            all(sum(len(p.text) for p in b) <= 7 and len(b) <= 2 for b in batches)
        )
        values = [[part.text.upper() for part in batch] for batch in batches]
        result = batch_results(texts, parts, batches, values)
        self.assertEqual(result, ["AAAAAAA\nAAAAA", "BBBB", "CCCCC", "D"])

    def test_bad_item_does_not_erase_other_sources(self):
        parts = [TextPart(0, "a"), TextPart(1, "b"), TextPart(2, "c")]
        values = join_parts(["a", "b", "c"], parts, ["А", None, "В"])
        self.assertEqual(values[0], "А")
        self.assertIsInstance(values[1], ValueError)
        self.assertEqual(values[2], "В")
        with self.assertRaises(ValueError):
            join_parts(["a"], [TextPart(2, "a")], ["А"])
        with self.assertRaises(ValueError):
            join_parts(["a"], parts, ["А"])

    def test_bad_batch_only_marks_its_own_sources(self):
        texts = ["a", "b"]
        parts = text_parts(texts, 3)
        result = batch_results(texts, parts, [[parts[0]], [parts[1]]], [[], ["Б"]])
        self.assertIsInstance(result[0], ValueError)
        self.assertEqual(result[1], "Б")

    def test_expired_budget_keeps_completed_work(self):
        cancel = Event()

        def work(value):
            if value == 0:
                cancel.set()
                return "ready"
            return value

        with ThreadPoolExecutor(1) as pool:
            values = map_bounded(pool, work, [0, 1, 2], RequestBudget(cancel), 1)
        self.assertEqual(values[0], "ready")
        self.assertTrue(
            all(isinstance(value, TranslationCancelled) for value in values[1:])
        )

    def test_cancel_returns_without_waiting_for_inflight_request(self):
        entered, release = Event(), Event()
        pool = ThreadPoolExecutor(1)

        def work(value):
            entered.set()
            release.wait(2)
            return value

        caller = ThreadPoolExecutor(1)
        try:
            future = caller.submit(
                map_bounded,
                pool,
                work,
                [1, 2, 3],
                RequestBudget(deadline=perf_counter() + 0.15),
                1,
            )
            self.assertTrue(entered.wait(1))
            result = future.result(timeout=0.7)
            self.assertTrue(all(isinstance(value, TimeoutError) for value in result))
        finally:
            release.set()
            pool.shutdown(wait=True)
            caller.shutdown(wait=True)

    def test_budgets_validate_and_propagate_stop(self):
        for value in (0, -1, float("nan"), float("inf"), True, "15"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_timeout(value)
        self.assertEqual(validate_timeout(2), 2.0)
        with self.assertRaises(ValueError):
            RequestBudget(deadline=float("nan"))
        with self.assertRaises(TimeoutError):
            RequestBudget(deadline=perf_counter() - 1).check()
        stop = Event()
        stop.set()
        with self.assertRaises(TranslationCancelled):
            RequestBudget(stop=stop).check()

    def test_retry_after_numeric_date_and_invalid_values(self):
        self.assertEqual(retry_delay("7", 5), 7)
        self.assertEqual(retry_delay("-7", 5), 5)
        self.assertEqual(retry_delay("999999", 5), 86400)
        for value in (None, "garbage", "nan", "inf"):
            self.assertEqual(retry_delay(value, 5), 5)
        value = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=90))
        self.assertGreater(retry_delay(value, 5), 85)

    def test_service_cooldown_survives_next_capture(self):
        state = ServiceCooldown("test")
        response = httpx.Response(
            429,
            headers={"Retry-After": "60"},
            request=httpx.Request("POST", "https://test.invalid"),
        )
        error = httpx.HTTPStatusError(
            "limited", request=response.request, response=response
        )
        state.record(error)
        with self.assertRaises(RuntimeError):
            state.check()
        self.assertGreater(state._until, perf_counter() + 55)


if __name__ == "__main__":
    unittest.main()
