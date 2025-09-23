import os, time, json, threading, math

# Simple thread-safe performance counters and timers with EWMA.
# Updated by probes; consumed by monitor script.

_LOCK = threading.RLock()
_STATE = {
    'start_time': time.time(),
    'last_flush': 0.0,
    'counters': {
        'frames_pgie': 0,
        'detections': 0,
        'frames_sgie': 0,
        'recognition_attempts': 0,
        'recognition_matches': 0,
        'faiss_searches': 0,
    },
    'timers': {
        'pgie_ms_ewma': 0.0,
        'sgie_ms_ewma': 0.0,
        'faiss_ms_ewma': 0.0,
        'embed_ms_ewma': 0.0,
    },
    'config': {
        'ewma_alpha': 0.2,
        'flush_interval': 2.0,
        'stats_path': os.environ.get('PERF_STATS_PATH', '/dev/shm/edge-deepstream/perf_stats.json'),
        'print_interval': 10.0,
        'last_print': 0.0,
        'verbose_level': int(os.environ.get('PERF_VERBOSE','0'))  # 0 quiet,1 summary,2 attach+summary
    }
}

os.makedirs(os.path.dirname(_STATE['config']['stats_path']), exist_ok=True)

def _ewma(key: str, ms: float):
    a = _STATE['config']['ewma_alpha']
    cur = _STATE['timers'][key]
    if cur <= 0.0:
        _STATE['timers'][key] = ms
    else:
        _STATE['timers'][key] = (1-a)*cur + a*ms


def incr(counter: str, delta: int = 1):
    with _LOCK:
        _STATE['counters'][counter] = _STATE['counters'].get(counter, 0) + delta


def time_block(timer_key: str):
    class _Ctx:
        def __enter__(self):
            self.t0 = time.time(); return self
        def __exit__(self, exc_type, exc, tb):
            dt_ms = (time.time() - self.t0) * 1000.0
            with _LOCK:
                _ewma(timer_key, dt_ms)
    return _Ctx()


def record(timer_key: str, dt_ms: float):
    with _LOCK:
        _ewma(timer_key, dt_ms)


def snapshot():
    with _LOCK:
        now = time.time()
        elapsed = max(1e-6, now - _STATE['start_time'])
        c = dict(_STATE['counters'])
        t = dict(_STATE['timers'])
        # Rates per second
        rates = {
            'fps_pgie': c['frames_pgie'] / elapsed,
            'fps_sgie': c['frames_sgie'] / elapsed,
            'detections_per_s': c['detections'] / elapsed,
            'recognitions_per_s': c['recognition_matches'] / elapsed,
            'faiss_searches_per_s': c['faiss_searches'] / elapsed,
        }
        return {'time': now, 'elapsed': elapsed, 'counters': c, 'timers': t, 'rates': rates}


def maybe_flush(force: bool=False):
    with _LOCK:
        now = time.time()
        if force or (now - _STATE['last_flush']) >= _STATE['config']['flush_interval']:
            snap = snapshot()
            try:
                tmp = _STATE['config']['stats_path'] + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(snap, f)
                os.replace(tmp, _STATE['config']['stats_path'])
                _STATE['last_flush'] = now
            except Exception:
                pass
        # Optional console summary
        try:
            if _STATE['config']['verbose_level'] >= 1 and (now - _STATE['config']['last_print']) >= _STATE['config']['print_interval']:
                s = snap if 'snap' in locals() else snapshot()
                r = s['rates']; t = s['timers']
                print(f"[PERF] pgie_fps={r['fps_pgie']:.2f} sgie_fps={r['fps_sgie']:.2f} det/s={r['detections_per_s']:.2f} rec/s={r['recognitions_per_s']:.2f} faiss_ms={t['faiss_ms_ewma']:.2f}")
                _STATE['config']['last_print'] = now
        except Exception:
            pass

