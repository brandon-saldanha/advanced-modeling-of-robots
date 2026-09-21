from lab_amoro.parallel_robot import *
from lab_amoro.plot_tools import * 
from biglide_models import *  # Modify this to use the biglide
import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

## =======================================
#Singularity Part


ti = 0  # example initial time
ts = 1 # example time at singularity
tf = 3.0  # example final time

P = np.array([
    [ti**8, ti**7, ti**6, ti**5, ti**4, ti**3, ti**2, ti, 1],
    [8*ti**7, 7*ti**6, 6*ti**5, 5*ti**4, 4*ti**3, 3*ti**2, 2*ti, 1, 0],
    [56*ti**6, 42*ti**5, 30*ti**4, 20*ti**3, 12*ti**2, 6*ti, 2, 0, 0],
    
    [ts**8, ts**7, ts**6, ts**5, ts**4, ts**3, ts**2, ts, 1],
    [8*ts**7, 7*ts**6, 6*ts**5, 5*ts**4, 4*ts**3, 3*ts**2, 2*ts, 1, 0],
    [56*ts**6, 42*ts**5, 30*ts**4, 20*ts**3, 12*ts**2, 6*ts, 2, 0, 0],
    
    [tf**8, tf**7, tf**6, tf**5, tf**4, tf**3, tf**2, tf, 1],
    [8*tf**7, 7*tf**6, 6*tf**5, 5*tf**4, 4*tf**3, 3*tf**2, 2*tf, 1, 0],
    [56*tf**6, 42*tf**5, 30*tf**4, 20*tf**3, 12*tf**2, 6*tf, 2, 0, 0]
])

xi = 0.0       # position at initial time from geometry of singularity
xs = 1.0       # position at singularity from geometry of singularity
xsD = -0.01   # velocity at singularity from trial and error
xsDD = -6.8e-4 # accelaration at singularity. Got it from the lecture book
xf = 2.0       # position at final time

cx = np.array([xi, 0, 0, xs, xsD, xsDD, xf, 0, 0])

# Solve for the coefficients a
a = np.linalg.solve(P, cx)

# Define functions for x(t), dot_x(t), and ddot_x(t)
def x(t):
    return (a[0] * t**8 + a[1] * t**7 + a[2] * t**6 +
            a[3] * t**5 + a[4] * t**4 + a[5] * t**3 +
            a[6] * t**2 + a[7] * t + a[8])

def xD(t):
    return (8 * a[0] * t**7 + 7 * a[1] * t**6 + 6 * a[2] * t**5 +
            5 * a[3] * t**4 + 4 * a[4] * t**3 + 3 * a[5] * t**2 +
            2 * a[6] * t + a[7])

def xDD(t):
    return (56 * a[0] * t**6 + 42 * a[1] * t**5 + 30 * a[2] * t**4 +
            20 * a[3] * t**3 + 12 * a[4] * t**2 + 6 * a[5] * t +
            2 * a[6])

## =======================================




def st(t, tf):
    return 10*(t/tf)**3 - 15*(t/tf)**4 + 6*(t/tf)**5

def stD(t, tf):
    return 30/tf*(t/tf)**2 - 60/tf*(t/tf)**3 + 30/tf*(t/tf)**4

def stDD(t, tf):
    return 60/(tf**2)*(t/tf) - 180/(tf**2)*(t/tf)**2 + 120/(tf**2)*(t/tf)**3		


def trajectory_generation(A, B, tf):
    dt = 0.01
    Ax = A[0]
    Bx = B[0]
    Ay = A[1]
    By = B[1]
		
    t = np.arange(0, tf + dt, dt).astype(float)
	
    xt = Ax + st(t, tf)*(Bx - Ax)
    yt = Ay + st(t, tf)*(By - Ay)
    Xt = np.vstack([xt, yt])
	
    xtD = stD(t, tf)*(Bx - Ax)
    ytD = stD(t, tf)*(By - Ay)
    XtD = np.vstack([xtD, ytD])
	
    xtDD = stDD(t, tf)*(Bx - Ax)
    ytDD = stDD(t, tf)*(By - Ay)
    XtDD = np.vstack([xtDD, ytDD])
    return Xt.T, XtD.T, XtDD.T
	
def joint_trajectory(A, B, tf):
    Xt, XtD, XtDD = trajectory_generation(A, B, tf)
    x = Xt[:, 0]
    y = Xt[:, 1]
	
    Vx = XtD[:, 0]
    Vy = XtD[:, 1]
	
    ax = XtDD[:, 0]
    ay = XtDD[:, 1]
	
    gamma1 = -1 
    gamma2 = -1
    assembly_mode = -1
	
    q11 = np.zeros((x.shape[0], 1))
    q21 = np.zeros((x.shape[0], 1))
    q12 = np.zeros((x.shape[0], 1))
    q22 = np.zeros((x.shape[0], 1))
    q11D = np.zeros((x.shape[0], 1))
    q21D = np.zeros((x.shape[0], 1))
    q12D = np.zeros((x.shape[0], 1))
    q22D = np.zeros((x.shape[0], 1))
    q11DD = np.zeros((x.shape[0], 1))
    q21DD = np.zeros((x.shape[0], 1))

    for i in range(x.shape[0]):
        q11[i], q21[i] = igm(x[i], y[i], gamma1, gamma2)
        q12[i], q22[i] = dgm_passive(q11[i][0], q21[i][0], assembly_mode)
        q11D[i], q21D[i] = ikm(q11[i][0], q12[i][0], q21[i][0], q22[i][0], Vx[i], Vy[i])
        q12D[i], q22D[i] = dkm_passive(q11[i][0], q12[i][0], q21[i][0], q22[i][0], q11D[i][0], q21D[i][0], Vx[i], Vy[i])
        q11DD[i], q21DD[i] = ikm2(q11[i][0], q12[i][0], q21[i][0], q22[i][0], q11D[i][0], q12D[i][0], q21D[i][0], q22D[i][0], ax[i], ay[i])
		
    return q11, q21, q11D, q21D, q11DD, q21DD

def plot_errors(time, joint1_errors, joint2_errors):
    plt.figure()
    plt.plot(time, joint1_errors, label='Joint 1 Error')
    plt.plot(time, joint2_errors, label='Joint 2 Error')
    plt.title("Joint Errors")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()

	# Set scientific notation for y-axis
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Force scientific notation

    plt.show()

def main(args=None):
    # Initialize and start the ROS2 robot interface
    rclpy.init(args=args)
    robot = Robot("biglide")  # Modify this to use the biglide
    start_robot(robot)

    # Prepare plots
    app = QtGui.QApplication([])
    scope_joint1 = Scope("Joint 1", -0.5, 1.5)
    scope_joint2 = Scope("Joint 2", -1.5, 1.5)

    # Create the trajectory as arrays in Cartesian space (position, velocity, acceleration)
    A = np.array([0., 0.])		
    B = np.array([-0.06, 1.4])
    tf = 2
	
    # Create the trajectory as arrays in joint space using the inverse models (position, velocity, acceleration)
    q11, q21, q11D, q21D, q11DD, q21DD = joint_trajectory(A, B, tf)
	
    qt = np.hstack([q11, q21])
    qtD = np.hstack([q11D, q21D])
    qtDD = np.hstack([q11DD, q21DD])

    index = 0

    # Prepare lists to store errors
    errors = {
        "time": [],
        "joint1": [],
        "joint2": [],
    }

    # Controller
    try:
        robot.apply_efforts(0.0, 0.0)  # Required to start the simulation
        while True:
            if robot.data_updated():
                # Robot available data
                q11 = robot.active_left_joint.position
                q21 = robot.active_right_joint.position
                q11D = robot.active_left_joint.velocity
                q21D = robot.active_right_joint.velocity
				
                assembly_mode = -1
                q12, q22 = dgm_passive(q11, q21, assembly_mode)
                xD, yD = dkm(q11, q12, q21, q22, q11D, q21D)
                q12D, q22D = dkm_passive(q11, q12, q21, q22, q11D, q21D, xD, yD)

                # CTC controller
                qa = np.array([q11, q21])
                qaD = np.array([q11D, q21D])
                M, c = dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D)

                kD = np.ones((2, 2)) * 2
                kP = np.ones((2, 2)) * 4
                a = qtDD[index, :] + kD@(qtD[index, :] - qaD) + kP@(qt[index, :] - qa)

                tau = M @ a + c
				
                tau_left = tau[0]
                tau_right = tau[1]

                robot.apply_efforts(tau_left, tau_right)

                # Scope update
                time = robot.get_time()
                if time < 5.0:
                    scope_joint1.update(time, qt[index, 0], qa[0])
                    scope_joint2.update(time, qt[index, 1], qa[1])

                    # Collect errors
                    errors["time"].append(time)
                    errors["joint1"].append(qt[index, 0] - qa[0])
                    errors["joint2"].append(qt[index, 1] - qa[1])

                if index < len(qt) - 1:
                    index += 1  # Next point in trajectory

    except KeyboardInterrupt:
        pass

    # Plot errors
    # plot_errors(errors["time"], errors["joint1"], errors["joint2"])

if __name__ == "__main__":
    main(sys.argv)
