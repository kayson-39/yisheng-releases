"""Read-only diagnostics use synthetic delivery times; no recorder or models."""
import queue
import unittest
from unittest.mock import patch

import numpy as np
from jpsub.core import pipeline


class CaptureClockTests(unittest.TestCase):
    def setUp(self):
        self.job = pipeline.Pipeline({})
        self.base = self.job._performance_started

    def tearDown(self):
        self.job._stop.set()
        self.job._shutdown_pools()
        self.job._release_resources()

    def deliver(self, at, samples):
        with patch.object(pipeline.time, 'monotonic', return_value=self.base + at):
            self.job._on_audio(np.zeros(samples, dtype=np.float32), 0.0)

    def sample(self, at):
        with patch.object(pipeline, '_write_performance') as writer:
            self.job._report_performance(self.base + at, 0, 0, 0)
        writer.assert_called_once()
        return writer.call_args.args[0]

    def test_no_callback_does_not_claim_zero_gap_or_age(self):
        stats = self.sample(30)
        self.assertNotIn('capture_clock_gap_s', stats)
        self.assertNotIn('capture_callback_age_s', stats)
        self.assertEqual(stats['audio_overload_dropped_samples'], 0)

    def test_clock_gap_excludes_startup_and_first_chunk(self):
        self.deliver(10, 32000)  # Timestamp is the end of this chunk, not its start.
        first = self.sample(30)
        self.assertEqual(first['capture_callback_age_s'], 20.0)
        self.assertNotIn('capture_clock_gap_s', first)
        self.deliver(42, 16000)
        later = self.sample(60)
        self.assertEqual(later['capture_clock_gap_s'], 31.0)
        self.assertEqual(later['capture_callback_age_s'], 18.0)

    def test_bursty_delivery_can_produce_negative_gap(self):
        self.deliver(1, 4000)
        self.deliver(1.5, 16000)
        self.assertEqual(self.sample(30)['capture_clock_gap_s'], -0.5)

    def test_pause_counts_delivery_without_enqueuing_and_stop_ignores_late_callback(self):
        self.deliver(1, 16000)
        self.job.pause(True)
        self.deliver(2, 16000)
        self.assertEqual(self.job._queue.qsize(), 1)
        self.assertEqual(self.job._captured_samples, 16000)  # Existing subtitle timeline.
        self.job._stop.set()
        self.deliver(3, 16000)
        stats = self.sample(30)
        self.assertEqual(stats['capture_clock_gap_s'], 0.0)
        self.assertEqual(stats['capture_callback_age_s'], 28.0)

    def test_overload_counter_survives_queue_drain_and_counts_only_discarded_audio(self):
        self.job._queue = queue.Queue(maxsize=2)
        self.deliver(1, 500)
        self.deliver(2, 300)
        self.deliver(3, 700)
        kept, _, position = self.job._queue.get_nowait()
        self.assertEqual((kept.size, position), (700, 800))
        self.deliver(4, 200)
        self.deliver(5, 100)
        self.deliver(6, 400)
        self.job._queue.get_nowait()
        stats = self.sample(30)
        self.assertEqual(stats['audio_queue'], 0)
        self.assertEqual(stats['audio_overload_events'], 2)
        self.assertEqual(stats['audio_overload_dropped_samples'], 1100)


if __name__ == '__main__':
    unittest.main()
