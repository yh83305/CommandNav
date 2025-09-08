## Instructions
---
A voice-guided robot uses **Detic** (open-vocabulary) to record image semantics during mapping, **Whisper** for speech recognition, **BERT** for semantic understanding, and **CLIP** for calculating image-text similarity to plan navigation paths.

## Environment Setup Process
---

### 1. Install ROS Noetic
```bash
wget http://fishros.com/install -O fishros && . fishros
```

### 2. Install Navigation Packages
```bash
sudo apt-get install ros-noetic-navigation
sudo apt-get install ros-noetic-gmapping
```

### 3. Create a Workspace and Clone the Repository
```bash
cd /home/ubuntu/Desktop/catkin_turtlebot3/src
catkin_init_workspace
git clone https://github.com/yh83305/CommandNav.git
cd ..
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

### 4. Configure the `source` File
```bash
sudo gedit ~/.bashrc
```
Add the following line:
```bash
source /home/ubuntu/Desktop/catkin_turtlebot3/devel/setup.bash
```
Then run:
```bash
source ~/.bashrc
```

### 5. Install Miniconda and Set Up the Python Environment
```bash
conda create -n rosp python=3.8
conda activate rosp
pip install rospkg rospy catkin_tools numpy defusedxml
pip install transformers torch sentence_transformers matplotlib
```

### 6. Install CLIP
```bash
cd /home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/CLIP-main
python setup.py build install
```

### 7. Install PyAudio
```bash
sudo apt-get install python-dev portaudio19-dev
pip install pyaudio
```

### 8. Install Whisper
```bash
pip install -U openai-whisper
```

### 9. Download and Configure BERT
- Download `pytorch_model.bin` from:
[https://huggingface.co/google-bert/bert-base-uncased/tree/main](https://huggingface.co/google-bert/bert-base-uncased/tree/main)
- Place it in:
`send_goals/src/BERT/bert-base-uncased`
- Train the model using `bert.py` (the trained model will be saved as `train_furniture2.pth`).

**Note:**
- Both models (BERT & Whisper) are called in `send_goals/src/nlu.py` (modify as needed).
- Update absolute paths in the code under `send_goals/src`.

### 10. Launch Semantic Navigation
```bash
roslaunch send_goals clip_nav.launch
```

(Translates Chinese commands into English for navigation.)
