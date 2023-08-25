import sys
import time
import math
import sympy as sp
from sympy import nsimplify
from sympy.physics.mechanics import dynamicsymbols
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d
#%matplotlib inline
#import sympy
sys.path.append('../lib/python/amd64')
import robot_interface as sdk


if __name__ == '__main__':

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    motiontime = 0
    d2 = {'FR_':0, 'FL_':1, 'RR_':2, 'RL_':3}
    d3 = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }

    #theta 1: hip joint angle, theta2: thigh joint angle, theta3: calf joint angle
    #theta, alpha, r, d D-H parameters
    theta1, theta2, theta3, hox, hoy, toy, l, theta, alpha, r, d = dynamicsymbols(
        'theta1 theta2 theta3 hox hoy toy l theta alpha r d')

    
    # Defining Homogenous Transformation matrix using D-H parameters
    rot = sp.Matrix([[sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.sin(alpha)],
                 [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha)],
                 [0, sp.sin(alpha), sp.cos(alpha)]])

    trans = sp.Matrix([r*sp.cos(theta), r*sp.sin(theta), d])

    last_row = sp.Matrix([[0, 0, 0, 1]])

    m = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    # Creating homogenous matrices
    # 1st is a translation from body frame to the translated frame ht
    b_T_ht = m.subs({theta:0, alpha:0, r:hox, d:0})

    #To translate rear joints
    b_T_ht_R = m.subs({theta:math.pi, alpha:0, r:hox, d:0})
    
    # 2nd is a translation from ht frame to hip joint frame
    ht_T_h = m.subs({theta:-math.pi/2, alpha:-math.pi/2, r:hoy, d:0})
    ht_T_h = nsimplify(ht_T_h, tolerance=1e-10, rational=True)

    # 2nd translation For FL
    ht_T_h_FL = m.subs({theta:math.pi/2, alpha:math.pi/2, r:hoy, d:0})
    ht_T_h_FL = nsimplify(ht_T_h_FL, tolerance=1e-10, rational=True)

    # 2nd translation for RR and RL
    ht_T_h_RR = m.subs({theta:math.pi/2, alpha:-math.pi/2, r:hoy, d:0})
    ht_T_h_RL = m.subs({theta:-math.pi/2, alpha:math.pi/2, r:hoy, d:0})

    #3rd is a rotation and a translation from hip joint to thigh joint location
    tt_T_h = m.subs({theta:theta1, alpha:0, r:toy, d:0})
    tt_T_h = nsimplify(tt_T_h, tolerance=1e-10, rational=True)

    #4th is transformation between the previous frame and the designated thigh joint frame
    t_T_tt = m.subs({theta:math.pi/2, alpha:-math.pi/2, r:0, d:0})
    t_T_tt = nsimplify(t_T_tt, tolerance=1e-10, rational=True)

    # 4th translation for FL
    t_T_tt_FL = m.subs({theta:-math.pi/2, alpha:-math.pi/2, r:0, d:0})
    t_T_tt_FL = nsimplify(t_T_tt_FL, tolerance=1e-10, rational=True)

    #5th is transforamtion from calf to thigh
    t_T_c = m.subs({theta:theta2, alpha:0, r:l, d:0})
    t_T_c = nsimplify(t_T_c, tolerance=1e-10, rational=True)

    #6th and final transformation is from calf joint to foot joint
    f_T_c = m.subs({theta:theta3, alpha:0, r:l, d:0})
    f_T_c = nsimplify(f_T_c, tolerance=1e-10, rational=True)

    # To get foot position w.r.t body frame we multiply all the homogenous transforms
    f_T_b = sp.simplify(b_T_ht * ht_T_h * tt_T_h * t_T_tt * t_T_c * f_T_c)
    # For FL
    f_T_b_FL = sp.simplify(b_T_ht * ht_T_h_FL * tt_T_h * t_T_tt_FL * t_T_c * f_T_c)
    # For RR
    f_T_b_RR = sp.simplify(b_T_ht_R * ht_T_h_RR * tt_T_h * t_T_tt * t_T_c * f_T_c)
    # For RL
    f_T_b_RL = sp.simplify(b_T_ht_R * ht_T_h_RL * tt_T_h * t_T_tt_FL * t_T_c  * f_T_c )

    
    # Numerical Evaluation # FR
    px = f_T_b[0, 3]
    py = f_T_b[1, 3]
    pz = f_T_b[2, 3]
    fx = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), px, 'numpy')
    fy = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), py, 'numpy')
    fz = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), pz, 'numpy')

    # Numerical Evaluation # FL
    px_FL = f_T_b_FL[0, 3]
    py_FL = f_T_b_FL[1, 3]
    pz_FL = f_T_b_FL[2, 3]
    fx_FL = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), px_FL, 'numpy')
    fy_FL = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), py_FL, 'numpy')
    fz_FL = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), pz_FL, 'numpy')

    # Numerical Evaluation # RR
    px_RR = f_T_b_RR[0, 3]
    py_RR = f_T_b_RR[1, 3]
    pz_RR = f_T_b_RR[2, 3]
    fx_RR = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), px_RR, 'numpy')
    fy_RR = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), py_RR, 'numpy')
    fz_RR = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), pz_RR, 'numpy')

    # Numerical Evaluation # RL
    px_RL = f_T_b_RL[0, 3]
    py_RL = f_T_b_RL[1, 3]
    pz_RL = f_T_b_RL[2, 3]
    fx_RL = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), px_RL, 'numpy')
    fy_RL = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), py_RL, 'numpy')
    fz_RL = sp.lambdify((hox, hoy, toy, l, theta1, theta2, theta3), pz_RL, 'numpy')


    # Required offset values
    hox = 0.1881
    hoy = 0.0465
    toy = 0.08
    l = 0.213

    i=0
    theta_hip = []
    theta_thigh = []
    theta_calf = []
    # For FL
    theta_hip_FL = []
    theta_thigh_FL = []
    theta_calf_FL = []

    # For RR
    theta_hip_RR = []
    theta_thigh_RR = []
    theta_calf_RR = []

    # For RL
    theta_hip_RL = []
    theta_thigh_RL = []
    theta_calf_RL = []

    x_foot_hardware = []
    y_foot_hardware = []
    z_foot_hardware = []

    x_foot_hardware_FL = []
    y_foot_hardware_FL = []
    z_foot_hardware_FL = []

    x_foot_hardware_RR = []
    y_foot_hardware_RR = []
    z_foot_hardware_RR = []

    x_foot_hardware_RL = []
    y_foot_hardware_RL = []
    z_foot_hardware_RL = []
     
    while i<=20000:
        i+=1
        time.sleep(0.002)
        motiontime = motiontime + 1

        udp.Recv()
        udp.GetRecv(state)

        cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
        cmd.gaitType = 0
        cmd.speedLevel = 0
        cmd.footRaiseHeight = 0
        cmd.bodyHeight = 0
        cmd.euler = [0, 0, 0]
        cmd.velocity = [0, 0]
        cmd.yawSpeed = 0.0
        cmd.reserve = 0

        if(motiontime > 0 and motiontime < 1000):
            cmd.mode = 1
            cmd.euler = [-0.3, 0, 0]
        
        if(motiontime > 1000 and motiontime < 2000):
            cmd.mode = 1
            cmd.euler = [0.3, 0, 0]
        
        if(motiontime > 2000 and motiontime < 3000):
            cmd.mode = 1
            cmd.euler = [0, -0.2, 0]
        
        if(motiontime > 3000 and motiontime < 4000):
            cmd.mode = 1
            cmd.euler = [0, 0.2, 0]
        
        if(motiontime > 4000 and motiontime < 5000):
            cmd.mode = 1
            cmd.euler = [0, 0, -0.2]
        
        if(motiontime > 5000 and motiontime < 6000):
            cmd.mode = 1
            cmd.euler = [0.2, 0, 0]
        
        if(motiontime > 6000 and motiontime < 7000):
            cmd.mode = 1
            cmd.bodyHeight = -0.2
        
        if(motiontime > 7000 and motiontime < 8000):
            cmd.mode = 1
            cmd.bodyHeight = 0.1
        
        if(motiontime > 8000 and motiontime < 9000):
            cmd.mode = 1
            cmd.bodyHeight = 0.0
        
        if(motiontime > 9000 and motiontime < 11000):
            cmd.mode = 5
        
        if(motiontime > 11000 and motiontime < 13000):
            cmd.mode = 6
        
        if(motiontime > 13000 and motiontime < 14000):
            cmd.mode = 0
        
        if(motiontime > 14000 and motiontime < 18000):
            cmd.mode = 2
            cmd.gaitType = 2
            cmd.velocity = [0.4, 0] # -1  ~ +1
            cmd.yawSpeed = 2
            cmd.footRaiseHeight = 0.1
            # printf("walk\n")
        
        if(motiontime > 18000 and motiontime < 20000):
            cmd.mode = 0
            cmd.velocity = [0, 0]
        
        if(motiontime > 20000 and motiontime < 24000):
            cmd.mode = 2
            cmd.gaitType = 1
            cmd.velocity = [0.2, 0] # -1  ~ +1
            cmd.bodyHeight = 0.1
            # printf("walk\n")
               

        theta_hip.append(state.motorState[d3['FR_0']].q)
        theta_thigh.append(state.motorState[d3['FR_1']].q)
        theta_calf.append(state.motorState[d3['FR_2']].q)

        theta_hip_FL.append(state.motorState[d3['FL_0']].q)
        theta_thigh_FL.append(state.motorState[d3['FL_1']].q)
        theta_calf_FL.append(state.motorState[d3['FL_2']].q)

        theta_hip_RR.append(state.motorState[d3['RR_0']].q)
        theta_thigh_RR.append(state.motorState[d3['RR_1']].q)
        theta_calf_RR.append(state.motorState[d3['RR_2']].q)

        theta_hip_RL.append(state.motorState[d3['RL_0']].q)
        theta_thigh_RL.append(state.motorState[d3['RL_1']].q)
        theta_calf_RL.append(state.motorState[d3['RL_2']].q)


        x_foot_hardware.append(state.footPosition2Body[d2['FR_']].x)
        y_foot_hardware.append(state.footPosition2Body[d2['FR_']].y)
        z_foot_hardware.append(state.footPosition2Body[d2['FR_']].z)

        x_foot_hardware_FL.append(state.footPosition2Body[d2['FL_']].x)
        y_foot_hardware_FL.append(state.footPosition2Body[d2['FL_']].y)
        z_foot_hardware_FL.append(state.footPosition2Body[d2['FL_']].z)

        x_foot_hardware_RR.append(state.footPosition2Body[d2['RR_']].x)
        y_foot_hardware_RR.append(state.footPosition2Body[d2['RR_']].y)
        z_foot_hardware_RR.append(state.footPosition2Body[d2['RR_']].z)

        x_foot_hardware_RL.append(state.footPosition2Body[d2['RL_']].x)
        y_foot_hardware_RL.append(state.footPosition2Body[d2['RL_']].y)
        z_foot_hardware_RL.append(state.footPosition2Body[d2['RL_']].z)

        udp.SetSend(cmd)
        udp.Send()

    x_foot = np.array(fx(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip), np.array(theta_thigh), np.array(theta_calf)))
    y_foot = np.array(fy(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip), np.array(theta_thigh), np.array(theta_calf)))
    z_foot = np.array(fz(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip), np.array(theta_thigh), np.array(theta_calf)))

    #For FL
    x_foot_FL = np.array(fx_FL(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_FL), np.array(theta_thigh_FL), np.array(theta_calf_FL)))
    y_foot_FL = np.array(fy_FL(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_FL), np.array(theta_thigh_FL), np.array(theta_calf_FL)))
    z_foot_FL = np.array(fz_FL(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_FL), np.array(theta_thigh_FL), np.array(theta_calf_FL)))

    #For RR
    x_foot_RR = np.array(fx_RR(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_RR), np.array(theta_thigh_RR), np.array(theta_calf_RR)))
    y_foot_RR = np.array(fy_RR(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_RR), np.array(theta_thigh_RR), np.array(theta_calf_RR)))
    z_foot_RR = np.array(fz_RR(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_RR), np.array(theta_thigh_RR), np.array(theta_calf_RR)))

    #For RL
    x_foot_RL = np.array(fx_RL(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_RL), np.array(theta_thigh_RL), np.array(theta_calf_RL)))
    y_foot_RL = np.array(fy_RL(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_RL), np.array(theta_thigh_RL), np.array(theta_calf_RL)))
    z_foot_RL = np.array(fz_RL(0.1881, 0.04675, 0.08, 0.213, np.array(theta_hip_RL), np.array(theta_thigh_RL), np.array(theta_calf_RL)))   

    print('FR:' , mean_squared_error(x_foot, x_foot_hardware))
    print(mean_squared_error(y_foot, y_foot_hardware))
    print(mean_squared_error(z_foot, z_foot_hardware))

    print('FL:' , mean_squared_error(x_foot_FL, x_foot_hardware_FL))
    print(mean_squared_error(y_foot_FL, y_foot_hardware_FL))
    print(mean_squared_error(z_foot_FL, z_foot_hardware_FL))

    print('RR:' , mean_squared_error(x_foot_RR, x_foot_hardware_RR))
    print(mean_squared_error(y_foot_RR, y_foot_hardware_RR))
    print(mean_squared_error(z_foot_RR, z_foot_hardware_RR))

    print('RL:' , mean_squared_error(x_foot_RL, x_foot_hardware_RL))
    print(mean_squared_error(y_foot_RL, y_foot_hardware_RL))
    print(mean_squared_error(z_foot_RL, z_foot_hardware_RL))

    
    # creating an empty figure for plotting
    fig = plt.figure()
 
    # defining a sub-plot with 1x2 axis and defining
    # it as first plot with projection as 3D
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    
    ax.plot3D(x_foot, y_foot, z_foot, label='p_{kinematics}FR')
    ax.plot3D(x_foot_hardware,y_foot_hardware,z_foot_hardware, label = 'p_{hardware}FR')
    ax.set_xlabel('Foot X position (m)')
    ax.set_ylabel('Foot Y position (m)')
    ax.set_zlabel('Foot Z position (m)')
    plt.legend(loc="upper right")
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot3D(x_foot_FL, y_foot_FL, z_foot_FL, label='p_{kinematics}FL')
    ax.plot3D(x_foot_hardware_FL,y_foot_hardware_FL,z_foot_hardware_FL, label = 'p_{hardware}FL')
    ax.set_xlabel('Foot X position (m)')
    ax.set_ylabel('Foot Y position (m)')
    ax.set_zlabel('Foot Z position (m)')
    plt.legend(loc="upper right")

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot3D(x_foot_RR, y_foot_RR, z_foot_RR, label='p_{kinematics}RR')
    ax.plot3D(x_foot_hardware_RR,y_foot_hardware_RR,z_foot_hardware_RR, label = 'p_{hardware}RR')
    ax.set_xlabel('Foot X position (m)')
    ax.set_ylabel('Foot Y position (m)')
    ax.set_zlabel('Foot Z position (m)')
    plt.legend(loc="upper right")
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot3D(x_foot_RL, y_foot_RL, z_foot_RL, label='p_{kinematics}RL')
    ax.plot3D(x_foot_hardware_RL,y_foot_hardware_RL,z_foot_hardware_RL, label = 'p_{hardware}RL')

    
    ax.set_xlabel('Foot X position (m)')
    ax.set_ylabel('Foot Y position (m)')
    ax.set_zlabel('Foot Z position (m)')
    plt.legend(loc="upper right")

    plt.show()


#     FR: 2.320204906269043e-05
# 1.0496184152597027e-05
# 0.00011938760973514153
# FL: 2.309405374773589e-05
# 1.0580780805757764e-05
# 0.00011937059331932549
# RR: 2.336110030515203e-05
# 1.0561915331879988e-05
# 0.0001192785566576987
# RL: 2.3167342368657767e-05
# 1.0512617881365844e-05
# 0.00011925389410878045

