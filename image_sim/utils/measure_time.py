import time


def measure_time(start_time, end_time):
    elapsed_time = end_time - start_time
    final_time = time.strftime("%M:%S", time.gmtime(elapsed_time))
    final_time_sec = f"{elapsed_time:.5f}s ({int(elapsed_time * 1000)} ms)"
    print(f"Время выполнения: {final_time} | в миллисекундах: {final_time_sec}")

    return elapsed_time