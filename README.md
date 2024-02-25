# Quadruped_Locomotion
This repository contains implementation of Quadrupedal locomotion from formulation of kinematics and dynamics to deploying it on an actual robot. The robot we use is Unitree Go1 Edu, a low cost quadruped robot which is open source and can be deployed both, using python APIs and ROS.

Unitree Go1 Motion 



https://github.com/dhruvkm2402/Quadruped_Locomotion_Kinematic_ModelVerification-/assets/99369975/03f2fa53-0176-4c1d-a2c8-39c0be1f441d

If you find the work useful, kindly cite the following paper:
Mehta, D., Kosaraju, K., and Krovi, V., "Actively Articulated Wheeled Architectures for Autonomous Ground Vehicles - Opportunities and Challenges," SAE Technical Paper 2023-01-0109, 2023, https://doi.org/10.4271/2023-01-0109.

In order to operate the Uniree Go1 Edu robot - make sure to hang it initially using a harness to avoid any damage. Unitree Go1 by default comes with ROS/ROS2 enabled services. Details are given in the following link
https://github.com/unitreerobotics/unitree_legged_sdk

In order to connect to he robot and operate it using Python APIs, connect the Go1 Robot via Ethernet and follow the insructions below:
# Commands
Run this command after you plug in the Ethernet Port,
You can find an extra device ID. For example, enpxxx
ifconfig <br />
sudo ifconfig enpxxx down # enpxxx is your PC usb port <br />
sudo ifconfig enpxxx 192.168.123.162/ <br />
sudo ifconfig enpxxx up <br />
ping 192.168.123.161 <br />

After connecting, run the following command: <br />
python3 Quadruped_Kinematics/kinematic_DHP.py

This repo is under active development. 
