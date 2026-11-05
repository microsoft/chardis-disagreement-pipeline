import datetime as dt
import time

def get_time_and_elapsed_status(start_time:float) -> str:
    elapsed_time = dt.timedelta(seconds=time.perf_counter() - start_time)
    return f"{dt.datetime.now()} (Elapsed Time: {elapsed_time})"