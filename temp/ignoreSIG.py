import signal
import time

def signal_handler(sig, frame):
    print("SIGINT received, but ignoring it.")

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

while True:
    print("hello")
    time.sleep(1)