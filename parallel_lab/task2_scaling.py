
import multiprocessing as mp
import time
import math
import statistics

from harness import emit_verify_tag


# ============================================================
# CONFIGURATION
# ============================================================

STUDENT_ID = 230103284

P_LOG = mp.cpu_count()
P_PHYS = 6  # Ryzen 5 5500U: verify on your laptop

N_STRONG = 1_600_000
N0_WEAK = 250_000

STAGE_A_FRACTION = 0.05
STAGE_A_OPERATIONS = 12
STAGE_B_ITERATIONS = 35

WARMUP_RUNS = 1
TIMED_RUNS = 3

P_VALUES = list(range(1, P_LOG + 1))


# ============================================================
# STAGE A: SERIAL COMPUTATION
# ============================================================

def stage_a_kernel(N):
    """
    Serial computation in the parent process.
    Processes the first 5% of elements.
    Performs 12 transcendental operations per element.
    """

    count = max(1, int(N * STAGE_A_FRACTION))

    result = 0.0

    for i in range(count):

        x = (i + 1) / (count + 1)

        for _ in range(STAGE_A_OPERATIONS // 2):
            x = math.sin(x)
            x = math.sqrt(abs(x) + 1e-12)

        result += x

    return result


# ============================================================
# STAGE B: PARALLEL COMPUTATION
# ============================================================

def stage_b_worker(args):
    """
    Parallel computational kernel.

    Each element performs 35 iterations.
    Each iteration performs a multiply-add and sin operation.
    """

    start_idx, count = args

    a = 1.0000001
    b = 0.0000003

    result = 0.0

    for i in range(count):

        x = (start_idx + i + 1) * 0.000001

        for _ in range(STAGE_B_ITERATIONS):
            x = math.sin(x * a + b)

        result += x

    return result


# ============================================================
# CREATE WORK PARTITIONS
# ============================================================

def create_tasks(N, p):
    """
    Divide N elements evenly among p processes.
    """

    base = N // p
    remainder = N % p

    tasks = []
    start = 0

    for worker in range(p):

        count = base + (1 if worker < remainder else 0)

        tasks.append((start, count))

        start += count

    return tasks


# ============================================================
# EXECUTE ONE COMPLETE TRIAL
# ============================================================

def run_trial(pool, N, p):
    """
    Stage A runs serially in the parent.
    Stage B runs using an existing multiprocessing pool.

    Returns:
        total_time_ms
        stage_a_time_ms
    """

    total_start = time.perf_counter_ns()

    # Stage A: serial computation
    stage_a_start = time.perf_counter_ns()

    serial_result = stage_a_kernel(N)

    stage_a_end = time.perf_counter_ns()

    stage_a_time_ms = (
        stage_a_end - stage_a_start
    ) / 1_000_000

    # Stage B: parallel computation
    tasks = create_tasks(N, p)

    parallel_results = pool.map(stage_b_worker, tasks)

    # Consume results to ensure the computation is completed.
    checksum = serial_result + sum(parallel_results)

    total_end = time.perf_counter_ns()

    total_time_ms = (
        total_end - total_start
    ) / 1_000_000

    return total_time_ms, stage_a_time_ms, checksum


# ============================================================
# BENCHMARK ONE CONFIGURATION
# ============================================================

def benchmark(pool, N, p):

    # Warm-up runs
    for _ in range(WARMUP_RUNS):
        run_trial(pool, N, p)

    total_times = []
    stage_a_times = []

    # Timed runs
    for _ in range(TIMED_RUNS):

        total_ms, stage_a_ms, _ = run_trial(
            pool, N, p
        )

        total_times.append(total_ms)
        stage_a_times.append(stage_a_ms)

    return (
        statistics.median(total_times),
        statistics.median(stage_a_times)
    )


# ============================================================
# KARPF-FLATT METRIC
# ============================================================

def karp_flatt(T1, Tp, p):

    speedup = T1 / Tp

    efficiency = speedup / p

    if p == 1:
        e = None
    else:
        e = (
            (1 / speedup - 1 / p)
            / (1 - 1 / p)
        )

    return speedup, efficiency, e


# ============================================================
# PHASE 2A: STRONG SCALING
# ============================================================

def strong_scaling():

    print("\n========================================")
    print("PHASE 2A: STRONG SCALING")
    print("========================================")

    print(f"Fixed workload N = {N_STRONG:,}")

    results = {}
    stage_a_results = {}

    for p in P_VALUES:

        print(f"\nTesting {p} processes...")

        # Pool is created OUTSIDE the timed region.
        with mp.Pool(processes=p) as pool:

            total_ms, stage_a_ms = benchmark(
                pool,
                N_STRONG,
                p
            )

        results[p] = total_ms
        stage_a_results[p] = stage_a_ms

        print(f"Median time: {total_ms:.3f} ms")
        print(f"Stage A time: {stage_a_ms:.3f} ms")

    # Calculate speedup and efficiency.
    T1 = results[1]

    print("\nSTRONG SCALING RESULTS")

    print(
        f"{'P':>5} "
        f"{'Time(ms)':>14} "
        f"{'Speedup':>12} "
        f"{'Efficiency':>12} "
        f"{'Karp-Flatt':>12}"
    )

    for p in P_VALUES:

        speedup, efficiency, e = karp_flatt(
            T1,
            results[p],
            p
        )

        e_text = "N/A" if e is None else f"{e:.5f}"

        print(
            f"{p:>5} "
            f"{results[p]:>14.3f} "
            f"{speedup:>12.3f} "
            f"{efficiency:>12.3f} "
            f"{e_text:>12}"
        )

    return results, stage_a_results


# ============================================================
# PHASE 2B: WEAK SCALING
# ============================================================

def weak_scaling():

    print("\n========================================")
    print("PHASE 2B: WEAK SCALING")
    print("========================================")

    print(f"Base workload N0 = {N0_WEAK:,}")

    results = {}
    stage_a_results = {}

    for p in P_VALUES:

        N = p * N0_WEAK

        print(
            f"\nTesting {p} processes "
            f"with N = {N:,}..."
        )

        with mp.Pool(processes=p) as pool:

            total_ms, stage_a_ms = benchmark(
                pool,
                N,
                p
            )

        results[p] = total_ms
        stage_a_results[p] = stage_a_ms

        print(f"Median time: {total_ms:.3f} ms")
        print(f"Stage A time: {stage_a_ms:.3f} ms")

    T1 = results[1]

    print("\nWEAK SCALING RESULTS")

    print(
        f"{'P':>5} "
        f"{'N':>14} "
        f"{'Time(ms)':>14} "
        f"{'Alpha':>10} "
        f"{'Gustafson':>12} "
        f"{'Efficiency':>12}"
    )

    gustafson_results = {}

    for p in P_VALUES:

        N = p * N0_WEAK

        alpha = (
            stage_a_results[p] / results[p]
        )

        gustafson = p - alpha * (p - 1)

        # Weak scaling efficiency based on runtime.
        efficiency = T1 / results[p]

        gustafson_results[p] = gustafson

        print(
            f"{p:>5} "
            f"{N:>14,} "
            f"{results[p]:>14.3f} "
            f"{alpha:>10.5f} "
            f"{gustafson:>12.3f} "
            f"{efficiency:>12.3f}"
        )

    return results, stage_a_results, gustafson_results


# ============================================================
# AMDAHL'S LAW
# ============================================================

def amdahl_analysis(strong_results, stage_a_results):

    print("\n========================================")
    print("AMDAHL'S LAW ANALYSIS")
    print("========================================")

    T1 = strong_results[1]

    # Serial fraction measured at p = 1.
    s = stage_a_results[1] / T1

    # Theoretical maximum speedup.
    S_infinity = 1 / s

    # Theoretical speedup at maximum logical processors.
    theoretical_speedup = 1 / (
        s + (1 - s) / P_LOG
    )

    # Actual measured speedup.
    observed_speedup = (
        T1 / strong_results[P_LOG]
    )

    print(f"Serial fraction: {s:.6f}")
    print(f"Theoretical maximum speedup: {S_infinity:.3f}")
    print(
        f"Theoretical speedup at P={P_LOG}: "
        f"{theoretical_speedup:.3f}"
    )
    print(
        f"Observed speedup at P={P_LOG}: "
        f"{observed_speedup:.3f}"
    )

    return s, observed_speedup


# ============================================================
# SMT PERFORMANCE ANALYSIS
# ============================================================

def smt_analysis(strong_results):

    print("\n========================================")
    print("SMT / PHYSICAL CORE ANALYSIS")
    print("========================================")

    if P_PHYS >= P_LOG:

        print("No additional logical processors beyond physical cores.")

        return

    if P_PHYS + 1 not in strong_results:

        print("Not enough measurements for SMT comparison.")

        return

    # Marginal speedup gained by adding the last physical core.
    before = (
        strong_results[P_PHYS - 1]
        / strong_results[P_PHYS]
    )

    # Marginal speedup gained by adding the first SMT worker.
    after = (
        strong_results[P_PHYS]
        / strong_results[P_PHYS + 1]
    )

    if before != 0:

        drop = (
            (before - after) / before
        ) * 100

    else:
        drop = 0

    print(f"Physical cores: {P_PHYS}")
    print(f"Logical processors: {P_LOG}")

    print(
        f"Marginal speedup before SMT: {before:.4f}"
    )

    print(
        f"Marginal speedup after SMT: {after:.4f}"
    )

    print(
        f"Marginal speedup reduction: {drop:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("PHASE 2: STRONG & WEAK SCALING")
    print("========================================")

    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores: {P_PHYS}")
    print(f"Logical processors: {P_LOG}")

    # 1. Strong scaling
    strong_results, strong_stage_a = strong_scaling()

    # 2. Weak scaling
    weak_results, weak_stage_a, gustafson_results = (
        weak_scaling()
    )

    # 3. Amdahl analysis
    s, observed_speedup = amdahl_analysis(
        strong_results,
        strong_stage_a
    )

    # 4. SMT analysis
    smt_analysis(strong_results)

    # 5. Verification hash
    strong_speedup = (
        strong_results[1] / strong_results[P_LOG]
    )

    weak_speedup = gustafson_results[P_LOG]

    print("\n========================================")
    print("PHASE 2 VERIFICATION")
    print("========================================")

    verify = emit_verify_tag(
        task_id="PHASE2",
        student_seed=STUDENT_ID,
        primary_metric=strong_speedup,
        secondary_metric=weak_speedup
    )

    print(verify)

    print("\nPHASE 2 COMPLETED")


if __name__ == "__main__":
    mp.freeze_support()
    main()
