安装ROS-noetic
wget http://fishros.com/install -O fishros && . fishros
安装导航包
sudo apt-get install ros-noetic-navigation
sudo apt-get install ros-noetic-gmapping
创建工作空间，将仓库文件作为功能包clone到src
cd /home/ubuntu/Desktop/catkin_turtlebot3/src
catkin_init_workspace

cd ..
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

sudo  gedit ~/.bashrc
添加"source /home/ubuntu/Desktop/catkin_turtlebot3/devel/setup.bash"
source ~/.bashrc

conda create -n rosp python=3.8
conda activate rosp
pip install rospkg rospy catkin_tools numpy defusedxml
pip install transformers torch sentence_transformers matplotlip

cd /home/ubuntu/Desktop/catkin_turtlebot3/src/send_goals/src/CLIP-main
python setup.py build install

sudo apt-get install python-dev portaudio19-dev
pip install pyaudio

pip install -U openai-whisper

roslaunch send_goals clip_nav.launch
