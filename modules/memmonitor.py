import threading
import os
import time

class MemUsageMonitor(threading.Thread):
    stop_flag = False
    max_usage = 0
    total = -1

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print(f"Recording max memory usage...\n")
        # check if we're using a scoped-down GPU environment (pynvml does not listen to CUDA_VISIBLE_DEVICES)
        # so that we can measure memory on the correct GPU
        try:
            isinstance(int(os.environ["CUDA_VISIBLE_DEVICES"]), int)
            handle = pynvml.nvmlDeviceGetHandleByIndex(
                int(os.environ["CUDA_VISIBLE_DEVICES"]))
        except (KeyError, ValueError) as pynvmlHandleError:
            if os.getenv("SD_WEBUI_DEBUG",
                         'False').lower() in ('true', '1', 'y'):
                print("[MemMon][WARNING]", pynvmlHandleError)
                print(
                    "[MemMon][INFO]",
                    "defaulting to monitoring memory on gpu 0 (use CUDA_VISIBLE_DEVICES to override)"
                )
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        while not self.stop_flag:
            m = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.max_usage = max(self.max_usage, m.used)
            # print(self.max_usage)
            time.sleep(0.1)
        print(f"Stopped recording.\n")

    def read(self):
        return self.max_usage, self.total

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop_flag = True
        return self.max_usage, self.total

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    print(f"Unable to initialize NVIDIA management. No memory stats.")
    class MemUsageMonitor(threading.Thread):
        def start(self):
            return
        def stop(self):
            return
        def read(self):
            return 0, 0
        def read_and_stop(self):
            return self.read()
