import subprocess
import re


# Nvidia-smi gpu memory parsing.
# (This is a slightly modified version of https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused)

# Use the functions specified here to automatically select the best GPU available on your system.
# For further information see:
# - https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth

# Additionally, make sure to set the following environment variable:
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# TODO: check per_process_gpu_memory_fraction
# - (https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory)

def find_best_gpu():
    # Run nvidia-smi -L command to find all available GPUs
    cmd = "nvidia-smi -L"
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    output = output.decode("ascii")

    # We expect lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    available = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        available.append(int(m.group("gpu_id")))

    # Compute dict of GPU id to memory allocated on that GPU.
    # Run nvidia-smi -L command to get memory information
    cmd = "nvidia-smi"
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    output = output.decode("ascii")
    gpu_output = output[output.find("GPU Memory"):]
    # We expect lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    memory_dict = {gpu_id: 0 for gpu_id in available}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        memory_dict[gpu_id] += gpu_memory

    # Compute list of tuples of (allocated memory, gpu id)
    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in memory_dict.items()]
    memory_gpu_map = sorted(memory_gpu_map)

    # Get available GPUs
    gpus = memory_gpu_map
    print('Available GPUs: {0}'.format(gpus))

    # Get the best GPU
    best_gpu = int(gpus[0][1])  # the first element of the list is the GPU with the most free memory
    print('Selecting GPU: {:d}'.format(best_gpu))

    return best_gpu
