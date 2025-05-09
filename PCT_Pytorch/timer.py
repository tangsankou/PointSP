import time

class ProfileTimer:
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.steps = {}

    def start(self):
        self.start_time = time.time()
        print(f"[{self.name}] Timer started.")

    def record_step(self, step_name):
        if self.start_time is None:
            raise Exception("Timer was not started. Use the start() method to start the timer.")
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.steps[step_name] = elapsed_time
        print(f"[{self.name}] Step '{step_name}' recorded: {elapsed_time:.6f} seconds")

    def stop(self):
        if self.start_time is None:
            raise Exception("Timer was not started. Use the start() method to start the timer.")
        
        total_time = time.time() - self.start_time
        print(f"[{self.name}] Timer stopped.")
        print(f"[{self.name}] Total elapsed time: {total_time:.6f} seconds")
        print(f"[{self.name}] Step breakdown:")
        for step, elapsed in self.steps.items():
            print(f"    {step}: {elapsed:.6f} seconds")

    def reset(self):
        self.start_time = None
        self.steps = {}
        print(f"[{self.name}] Timer reset.")