"""Queueing simulation: fixed-entry vs cyclic-entry admission for layer-streamed DiT inference.

Weights stream in a fixed cyclic order, one block group per time slot (slot length constant, i.e. transfer-bound,
the most favorable case for cyclic entry). Requests arrive as a Poisson process, each needs N full forwards
(N * K block slots once admitted), batch capacity B. Fixed entry admits only when the stream is at block 1;
cyclic entry admits at any block. Reports throughput, mean batch occupancy and mean latency (in cycles).
"""

import numpy as np


def simulate(K: int, N: int, B: int, load: float, cyclic: bool, slots: int = 400_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rate = load * B / (N * K)  # arrivals per slot; load 1 = saturation
    queue, active, done = [], [], []  # queue: arrival slots; active: [arrival, remaining block slots]
    occupancy = 0
    for slot in range(slots):
        queue.extend([slot] * rng.poisson(rate))
        if cyclic or slot % K == 0:
            while queue and len(active) < B:
                active.append([queue.pop(0), N * K])
        occupancy += len(active)
        for request in active:
            request[1] -= 1
        done.extend(slot + 1 - a for a, r in active if r == 0)
        active = [r for r in active if r[1] > 0]
    warm = len(done) // 10
    latency = np.array(done[warm:]) / K
    return {"throughput_per_cycle": len(done) / slots * K, "mean_batch": occupancy / slots,
            "latency_cycles": float(latency.mean()), "queue_left": len(queue)}


if __name__ == "__main__":
    K, B = 4, 8
    print(f"K={K} block groups, batch capacity B={B}; latency in weight-streaming cycles (one cycle = K slots)")
    for N in (1, 4, 20):
        for load in (0.3, 0.7, 0.95):
            f, c = simulate(K, N, B, load, False), simulate(K, N, B, load, True)
            print(f"N={N:2d} load={load:.2f} | throughput/cycle fixed {f['throughput_per_cycle']:.3f} cyclic "
                  f"{c['throughput_per_cycle']:.3f} | mean batch fixed {f['mean_batch']:.2f} cyclic {c['mean_batch']:.2f}"
                  f" | latency fixed {f['latency_cycles']:.2f} cyclic {c['latency_cycles']:.2f}"
                  f" ({100 * (1 - c['latency_cycles'] / f['latency_cycles']):.1f}% lower)")
