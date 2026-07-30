import asyncio
import queue
from types import SimpleNamespace

from livepeer_gateway.decode_metrics_sim import (
    _actual_decoder_snapshot,
    simulate_decoder_metric_drift,
)


class TestDecodeMetricsSimulation:
    def test_decoder_metric_drift_stays_small_in_real_pyav_pipeline(self) -> None:
        report = asyncio.run(
            simulate_decoder_metric_drift(
                frame_count=90,
                producer_chunk_size=188 * 8,
                feed_delay_s=0.0005,
                consumer_delay_s=0.008,
                sample_interval_s=0.0005,
            )
        )

        assert report.decoded_frames > 0
        assert report.sample_count > 0
        assert report.max_abs_drift_queued_chunks <= 1
        assert report.max_abs_drift_queued_bytes <= report.producer_chunk_size
        assert report.max_abs_drift_buffered_bytes <= report.producer_chunk_size
        assert report.max_abs_drift_output_items_queued <= 1

        input_queue: queue.Queue[object] = queue.Queue()
        input_queue.put(b"abc")
        input_queue.put(object())
        output_queue: queue.Queue[object] = queue.Queue()
        output_queue.put(object())
        decoder = SimpleNamespace(
            _reader=SimpleNamespace(_queue=input_queue, _buffer=bytearray(b"de")),
            _output=output_queue,
        )
        assert _actual_decoder_snapshot(decoder) == (1, 3, 2, 1)
