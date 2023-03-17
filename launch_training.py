import os
import subprocess

current_working_dir = os.getcwd()
print("Current working directory: " + current_working_dir)

# Build the docker image
print("\nBuilding docker image...\n")
cmd_build_docker = "docker build -t pytorch:latest ."
error_code = subprocess.call(cmd_build_docker, shell=True)
if error_code != 0:
    print("Building of docker failed!")
    exit(error_code)

# Normalize dataset images to 224x224
print("\nNormalizing dataset images...")
cmd_normalize_dataset = "docker run --rm -it --gpus all --ipc=host -w /tmp -v " + current_working_dir + ":/tmp pytorch:latest python3 norm.py"
error_code = subprocess.call(cmd_normalize_dataset, shell=True)
if error_code != 0:
    print("Normalization of data failed!")
    exit(error_code)

# Split dataset into train and test
print("\nSplitting dataset...")
cmd_split_dataset = "docker run --rm -it --gpus all --ipc=host -w /tmp -v " + current_working_dir + ":/tmp pytorch:latest python3 split_dataset.py"
error_code = subprocess.call(cmd_split_dataset, shell=True)
if error_code != 0:
    print("Dataset split failed!")
    exit(error_code)

# Launch the training
print("\nLaunching training...")
cmd_training = "docker run --rm -it --gpus all --ipc=host -w /tmp -v " + current_working_dir + ":/tmp pytorch:latest python3 train.py"
error_code = subprocess.call(cmd_training, shell=True)
if error_code != 0:
    print("Training failed!")
    exit(error_code)