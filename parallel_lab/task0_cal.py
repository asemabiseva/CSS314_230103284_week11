
import multiprocessing as mp
import time

from harness import emit_verify_tag


STUDENT_ID = 230103284


def main():
    print("===== PHASE 0: HARDWARE CALIBRATION =====")

    # 1. Detect logical CPU count
    cpu_count = mp.cpu_count()

    print(f"Logical CPU count: {cpu_count}")

    # 2. Measure clock-read latency
    iterations = 2_000_000

    print(f"Clock read iterations: {iterations}")

    start = time.perf_counter_ns()

    for _ in range(iterations):
        time.perf_counter_ns()

    end = time.perf_counter_ns()

    total_time_ns = end - start
    average_latency_ns = total_time_ns / iterations

    print(f"Total clock read time: {total_time_ns:,} ns")
    print(f"Average clock read latency: {average_latency_ns:.4f} ns")

    # 3. Generate verification hash
    verify_output = emit_verify_tag(
        task_id="PHASE0",
        student_seed=STUDENT_ID,
        primary_metric=average_latency_ns,
        secondary_metric=float(cpu_count)
    )

    print(verify_output)

    print("===== PHASE 0 COMPLETED =====")


if __name__ == "__main__":
    main()
