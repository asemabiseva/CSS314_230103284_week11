
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
import statistics

from harness import run_benchmark, emit_verify_tag


# ============================================================
# CONFIGURATION
# ============================================================

STUDENT_ID = 230103284  # Replace with your actual student ID

M = 24_000_000
SHM_SIZE = 4096

STRIDES = [1, 2, 4, 8, 16, 32]

# Ryzen 5 5500U: 6 physical cores, 12 logical threads.
# Verify these values on your laptop.
P_PHYS = 6
P_LOG = mp.cpu_count()

P_VALUES = sorted(set(
    [1, 2, 4, P_PHYS, P_LOG]
))


# ============================================================
# WORKER FUNCTION
# ============================================================

def fs_worker(shm_name, slot_idx, iters):
    """
    Each process updates its own uint64 slot in shared memory.
    """

    shm = shared_memory.SharedMemory(name=shm_name)

    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)

    acc = arr[slot_idx]

    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005

    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask

    arr[slot_idx] = acc

    # Release exported buffer before closing shared memory.
    del arr
    shm.close()


# ============================================================
# FALSE SHARING EXPERIMENT
# ============================================================

def run_false_sharing(p, stride):

    shm = shared_memory.SharedMemory(
        create=True,
        size=SHM_SIZE
    )

    processes = []

    # Initialize shared memory.
    shm.buf[:] = bytes(SHM_SIZE)

    # Divide total iterations between workers.
    base_iters = M // p
    remainder = M % p

    try:
        for worker_id in range(p):

            iters = base_iters + (
                1 if worker_id < remainder else 0
            )

            slot_idx = worker_id * stride

            process = mp.Process(
                target=fs_worker,
                args=(
                    shm.name,
                    slot_idx,
                    iters
                )
            )

            processes.append(process)

        # Start all workers.
        for process in processes:
            process.start()

        # Wait for completion.
        for process in processes:
            process.join()

        # Detect worker failures.
        for process in processes:
            if process.exitcode != 0:
                raise RuntimeError(
                    f"Worker failed: exit code {process.exitcode}"
                )

    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()

        shm.close()
        shm.unlink()


# ============================================================
# BENCHMARK HELPER
# ============================================================

def benchmark_configuration(p, stride):

    stats = run_benchmark(
        lambda: run_false_sharing(p, stride),
        warmup_runs=2,
        timed_runs=7
    )

    return stats


# ============================================================
# TABLE 1A: STRIDE SENSITIVITY
# ============================================================

def stride_sweep():

    print("\n========== TABLE 1A: STRIDE SENSITIVITY ==========")

    results = {}

    print(
        f"{'Stride':>8} "
        f"{'Bytes':>8} "
        f"{'Median(ms)':>15} "
        f"{'IQR(ms)':>12} "
        f"{'Mops/s':>12}"
    )

    for stride in STRIDES:

        stats = benchmark_configuration(
            P_PHYS,
            stride
        )

        bytes_apart = stride * 8

        throughput = (
            M / (stats.median_ms / 1000)
        ) / 1_000_000

        results[stride] = stats

        print(
            f"{stride:>8} "
            f"{bytes_apart:>8} "
            f"{stats.median_ms:>15.3f} "
            f"{stats.iqr_ms:>12.3f} "
            f"{throughput:>12.3f}"
        )

    return results


# ============================================================
# KARPF-FLATT METRIC
# ============================================================

def karp_flatt(t1, tp, p):

    speedup = t1 / tp
    efficiency = speedup / p

    if p == 1:
        e = None
    else:
        e = (
            (1 / speedup) - (1 / p)
        ) / (
            1 - (1 / p)
        )

    return speedup, efficiency, e


# ============================================================
# TABLE 1B: PROCESS SCALING
# ============================================================

def process_scaling(stride):

    print(
        f"\n========== TABLE 1B: PROCESS SCALING "
        f"(STRIDE={stride}) =========="
    )

    results = {}

    # Measure all process counts.
    for p in P_VALUES:

        stats = benchmark_configuration(
            p,
            stride
        )

        results[p] = stats.median_ms

    # T1 is the measured time with one process.
    t1 = results[1]

    print(
        f"{'Processes':>12} "
        f"{'Time(ms)':>12} "
        f"{'Speedup':>12} "
        f"{'Efficiency':>12} "
        f"{'e(p)':>12}"
    )

    for p in P_VALUES:

        tp = results[p]

        speedup, efficiency, e = karp_flatt(
            t1,
            tp,
            p
        )

        e_text = "N/A" if e is None else f"{e:.5f}"

        print(
            f"{p:>12} "
            f"{tp:>12.3f} "
            f"{speedup:>12.3f} "
            f"{efficiency:>12.3f} "
            f"{e_text:>12}"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("==============================================")
    print("PHASE 1: FALSE SHARING & CACHE-LINE CONTENTION")
    print("==============================================")

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical processors: {P_LOG}")
    print(f"Physical cores: {P_PHYS}")
    print(f"Total iterations: {M}")

    # Experiment 1: stride sweep.
    stride_results = stride_sweep()

    # Experiment 2: process scaling.
    scaling_stride1 = process_scaling(1)
    scaling_stride16 = process_scaling(16)

    # Verification hash uses stride sweep medians
    # at maximum physical core count.
    median_stride1 = stride_results[1].median_ms
    median_stride16 = stride_results[16].median_ms

    print("\n========== VERIFICATION ==========")

    verify = emit_verify_tag(
        task_id="PHASE1",
        student_seed=STUDENT_ID,
        primary_metric=median_stride1,
        secondary_metric=median_stride16
    )

    print(verify)

    print("\n========== PHASE 1 COMPLETED ==========")


if __name__ == "__main__":
    mp.freeze_support()
    main()
