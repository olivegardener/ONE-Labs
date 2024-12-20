# Check system info
import sys
import platform
import subprocess

def get_mac_gpu_info():
    try:
        # Run system_profiler command
        cmd = ['system_profiler', 'SPDisplaysDataType']
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error getting GPU info: {e}"

print(f"System: {platform.system()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print("\nGPU Information:")
print(get_mac_gpu_info())
]