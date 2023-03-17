
# Training and Inference using Hugging Face ViT Model

Automatically launch the training of a ViT model using Hugging Face.
The python script for inference is also provided.


## Installation

prerequisites: Docker (https://docs.docker.com/engine/install/) and CUDA drivers (https://docs.nvidia.com/cuda/cuda-installation-guide-linux). \
\
The folder should be organized like indicated:

    .
    ├── Dockerfile
    ├── inference.py 
    ├── launch_training.py
    ├── norm.py
    ├── split_dataset.py
    ├── train.py
    ├── dataset                   
    │   ├── class1 # folder containing all the images of the class1
    │   ├── class2 # folder containing all the images of the class2
    │   └── class3 # folder containing all the images of the class3          
    │   └── ...
## Run Locally

Clone the project

```bash
  git clone https://github.com/tachillon/Training-and-Inference-Hugging-Face-ViT-Model.git
```

Go to the project directory

```bash
  cd Training-and-Inference-Hugging-Face-ViT-Model
```

Launch python script

```bash
  python3 launch_training.py
```

