"""Optional wall-time instrumentation; never supplies estimator evidence or source time."""

import ctypes
import json
import math
import os
import time
from pathlib import Path

import numpy as np


class RuntimeSource:
    """Measure read/group service cost and optionally pace original recorded arrivals.

    Speed >1 is transport load, not native higher-rate or rate-faithful imagery. File
    backpressure is retained: late groups are processed, never dropped. No image, stamp
    or source-time job budget changes. Runtime samples stay outside evidence diagnostics.
    """

    def __init__(self, source, output, *, transport_speed=None):
        if transport_speed is not None and (not math.isfinite(transport_speed) or transport_speed <= 0):
            raise ValueError('transport_speed must be finite and positive')
        self.source, self.output, self.speed = source, Path(output), transport_speed
        self.samples = []

    def __iter__(self):
        run_start = time.perf_counter_ns()
        iterator = iter(self.source)
        origin_clock = origin_arrival = None
        with self.output.open('x', encoding='utf-8') as handle:
            while True:
                read_start = time.perf_counter_ns()
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                ready = time.perf_counter_ns()
                arrival = max(image.arrival_ns for image in frame.images)
                if origin_clock is None:
                    origin_clock, origin_arrival = ready, arrival
                due = ready if self.speed is None else origin_clock + round((arrival - origin_arrival) / self.speed)
                if self.speed is not None:
                    remaining = due - time.perf_counter_ns()
                    if remaining > 0:
                        time.sleep(remaining / 1e9)
                start = time.perf_counter_ns()
                yield frame
                finished = time.perf_counter_ns()
                sample = {'record': 'RuntimeSample', 'group_id': frame.group_id,
                    't_ns': frame.t_ns, 'reader_ns': ready - read_start,
                    'group_service_ns': finished - start, 'pacing_wait_ns': start - ready,
                    'last_packet_to_completion_ns': finished - due}
                self.samples.append(sample)
                handle.write(json.dumps(sample) + '\n')
        self.elapsed_ns = time.perf_counter_ns() - run_start

    def trailing_events(self):
        return self.source.trailing_events()

    def calibrations(self):
        return self.source.calibrations()

    def summary(self, diagnostics):
        stalls = {}
        with Path(diagnostics).open(encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                if record.get('record') == 'SchedulerEvent' and record.get('kind') == 'stall':
                    group = record['group_id']
                    stalls[group] = stalls.get(group, 0) + record['stall_ns']

        def stats(values):
            values = np.asarray(values, np.float64) / 1e6
            return None if not values.size else {'mean_ms': float(values.mean()),
                'p95_ms': float(np.percentile(values, 95)), 'p99_ms': float(np.percentile(values, 99))}

        return {'groups': len(self.samples), 'transport_speed': self.speed,
            'profile': 'unpaced file replay' if self.speed is None else 'paced original-arrival transport replay',
            'steady_elapsed_seconds': self.elapsed_ns / 1e9,
            'elapsed_per_output_ms': self.elapsed_ns / max(len(self.samples), 1) / 1e6,
            'reader': stats([sample['reader_ns'] for sample in self.samples]),
            'group_service': stats([sample['group_service_ns'] for sample in self.samples]),
            'nonwaiting_pipeline': stats([max(0, sample['reader_ns'] + sample['group_service_ns']
                - stalls.get(sample['group_id'], 0)) for sample in self.samples]),
            'nonwaiting_pipeline_definition': 'reader plus complete group service, subtracting logged worker join stalls; includes handoff',
            'last_packet_to_completion': None if self.speed is None else stats([
                sample['last_packet_to_completion_ns'] for sample in self.samples]),
            'timing_limit': 'pacing uses constituent packet arrivals; deadline-closure wait and physical exposure latency are not instrumented',
            'rate_qualified': False,
            'qualification_limit': 'telemetry only; five-minute rate-faithful workload and queue gates are separate'}


def process_peak_memory():
    """Windows process peaks, including startup; None on other platforms."""
    if os.name != 'nt':
        return None

    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong), *[
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    if not psapi.GetProcessMemoryInfo(ctypes.c_void_p(kernel.GetCurrentProcess()),
                                    ctypes.byref(counters), ctypes.sizeof(counters)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'peak_working_set_bytes': counters.PeakWorkingSetSize,
            'peak_commit_bytes': counters.PeakPagefileUsage}
