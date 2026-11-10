"""Capture startup/error regressions; no device is opened and no model loaded."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jpsub.core import audio


class CaptureStartupTests(unittest.TestCase):
    def setUp(self):
        self.errors = []
        self.statuses = []
        self.capture = audio.AudioCapture(
            'loopback', '', Mock(), self.errors.append,
            on_status=self.statuses.append)
        self.device = SimpleNamespace(name='fake', samplerate=48000, recorder=Mock())

    def test_resampler_initialization_failure_reaches_pipeline_error_callback(self):
        with patch.object(audio, '_pick_device', return_value=self.device), patch.object(
                audio.soxr, 'ResampleStream', side_effect=RuntimeError('resampler failed')):
            self.capture.run()
        self.assertEqual(len(self.errors), 1)
        self.assertIn('resampler failed', self.errors[0])
        self.device.recorder.assert_not_called()

    def test_stop_before_thread_start_does_not_enumerate_or_open_device(self):
        self.capture.stop()
        with patch.object(audio, '_pick_device', side_effect=RuntimeError('must not enumerate')) as pick:
            self.capture.run()
        pick.assert_not_called()
        self.assertEqual(self.errors, [])
        self.assertEqual(self.statuses, [])

    def test_stop_during_device_selection_does_not_report_or_open_device(self):
        def select(*args):
            self.capture.stop()
            return self.device
        with patch.object(audio, '_pick_device', side_effect=select), patch.object(
                audio.soxr, 'ResampleStream') as resample:
            self.capture.run()
        self.assertEqual(self.errors, [])
        self.assertEqual(self.statuses, [])
        resample.assert_not_called()
        self.device.recorder.assert_not_called()

    def test_device_error_after_stop_is_not_reported_as_live_failure(self):
        def select(*args):
            self.capture.stop()
            raise RuntimeError('device disappeared')
        with patch.object(audio, '_pick_device', side_effect=select):
            self.capture.run()
        self.assertEqual(self.errors, [])

    def test_stop_during_resampler_initialization_does_not_open_device(self):
        def resample(*args, **kwargs):
            self.capture.stop()
            return Mock()
        with patch.object(audio, '_pick_device', return_value=self.device), patch.object(
                audio.soxr, 'ResampleStream', side_effect=resample):
            self.capture.run()
        self.device.recorder.assert_not_called()
        self.assertEqual(self.errors, [])

    def test_stop_during_recorder_enter_closes_without_read_or_ready_status(self):
        read, closed = Mock(), Mock()
        capture = self.capture
        class Recorder:
            def __enter__(self):
                capture.stop()
                return SimpleNamespace(record=read)
            def __exit__(self, *args):
                closed()
        self.device.recorder.return_value = Recorder()
        with patch.object(audio, '_pick_device', return_value=self.device), patch.object(
                audio.soxr, 'ResampleStream'):
            self.capture.run()
        read.assert_not_called()
        closed.assert_called_once()
        self.assertEqual(self.statuses, [])
        self.assertEqual(self.errors, [])

    def test_read_error_closes_recorder_and_notifies_owner(self):
        closed = Mock()
        class Recorder:
            def __enter__(self):
                return self
            def record(self, **kwargs):
                raise RuntimeError('read failed')
            def __exit__(self, *args):
                closed()
        self.device.recorder.return_value = Recorder()
        with patch.object(audio, '_pick_device', return_value=self.device), patch.object(
                audio.soxr, 'ResampleStream'):
            self.capture.run()
        closed.assert_called_once()
        self.assertEqual(self.statuses, ['正在采集：fake'])
        self.assertEqual(len(self.errors), 1)
        self.assertIn('read failed', self.errors[0])


if __name__ == '__main__':
    unittest.main()
