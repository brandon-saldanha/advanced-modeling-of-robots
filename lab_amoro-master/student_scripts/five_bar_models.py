from math import *
import numpy as np

# Geometric parameters
l = 0.09
d = 0.118

# Dynamic parameters
ZZ1R = 1.0 * 0.045 * 0.045
ZZ2R = 1.0 * 0.045 * 0.045
mR = 0.5

# Make functions for all the reapiting code
def get_A22H(q11, q21):
	A22H = 1/2*np.array([-l * cos(q21) - d + l*cos(q11), -l*sin(q21) + l * sin(q11)])
	return A22H

def get_A21A22(q21):
	A21A22 = np.array([l * cos(q21), l* sin(q21)])
	return A21A22

def dgm(q11, q21, assembly_mode):
	A22H = get_A22H(q11, q21)
	dA22H = np.linalg.norm(A22H)
	h = sqrt(l**2 - dA22H**2)
	HA13 = assembly_mode * (h / dA22H) * np.array([[0, -1], [1, 0]]) @ A22H
	OA21 = np.array([d/2., 0.])
	A21A22 = get_A21A22(q21)
	OA13 = OA21 + A21A22 + A22H + HA13
	
	x = OA13[0]
	y = OA13[1]
	return x, y

def get_A11A13(x, y, d):
	A11A13 = np.array([d/2 + x, y])
	return A11A13

def get_A11A12_IGM(x, y, d, gamma1):
	A11A13 = get_A11A13(x, y, d)
	A11M1 = 1/2 * A11A13
	c = np.linalg.norm(A11M1)
	b = sqrt(l**2 - c**2)
	M1A12 =  gamma1 * b/c * np.array([[0, -1], [1, 0]]) @ A11M1
	A11A12 = A11M1 + M1A12
	return A11A12

def get_A21A13(x, y, d):
	A21A13 = np.array([d/2 - x, y])
	return A21A13

def get_A21A22_IGM(x, y, d, gamma1):
	A21A13 = get_A21A13(x, y, d)
	A21M1 = 1/2 * A21A13
	c = np.linalg.norm(A21M1)
	b = sqrt(l**2 - c**2)
	M1A22 =  gamma1 * b/c * np.array([[0, -1], [1, 0]]) @ A21M1
	A21A22 = A21M1 + M1A22
	return A21A22

def igm(x, y, gamma1, gamma2): # wrong
	# Left arm
	A13 = np.array([[x],[y]])
	A11A12 = get_A11A12_IGM(x, y, d, gamma1) 
	q11 = atan2(A11A12[1], A11A12[0])
	
	# Right arm
	A12 = np.array([[d/2],[0]])
	
	A12A13 = A13 - A12
	A12M2 = 0.5 * A12A13
	c = np.linalg.norm(A12M2)
	b = sqrt(l**2 - c**2)
	M2A22 = gamma2 * b/c * np.array([[0, -1], [1, 0]]) @ A12M2
	
	A12A22 = A12M2 + M2A22
	q21 = atan2(A12A22[1], A12A22[0])
	return q11, q21

def dgm_passive(q11, q21, assembly_mode):
	x, y = dgm(q11, q21, assembly_mode)
	q12 = atan2(y/l - sin(q11), x/l + d/(2 * l) - cos(q11)) - q11
	q22 = atan2(y/l - sin(q21), x/l - d/(2 * l) - cos(q21)) - q21
	return q12, q22

def get_u11(q11):
	return np.array([cos(q11), sin(q11)])
def get_u12(q11, q12):
	return np.array([cos(q11 + q12), sin(q11 + q12)])
def get_u21(q21):
	return np.array([cos(q21), sin(q21)])
def get_u22(q21, q22):
	return np.array([cos(q21 + q22), sin(q21 + q22)])

def rotate_90_counterclockwise(vector):
	rotated_vector = np.zeros(2)
	rotated_vector[0] = vector[1]*-1
	rotated_vector[1] = vector[0]
	return rotated_vector

def get_v11(u11):
	return rotate_90_counterclockwise(u11)
def get_v12(u12):
	return rotate_90_counterclockwise(u12)
def get_v21(u21):
	return rotate_90_counterclockwise(u21)
def get_v22(u22):
	return rotate_90_counterclockwise(u22)

# You can create intermediate functions to avoid redundant code
def compute_A_B(q11, q12, q21, q22):
	u12 = get_u12(q11, q12)
	u22 = get_u22(q21, q22)
	v11 = get_v11(get_u11(q11))
	v21 = get_v21(get_u21(q21))
	A = np.vstack((u12.T, u22.T))
	B = np.vstack((
		np.array([l*np.dot(u12, v11), 0]), \
		np.array([0, l*np.dot(u22, v21)])
	))
	return A, B

def dkm(q11, q12, q21, q22, q11D, q21D):
	A, B = compute_A_B(q11, q12, q21, q22)
	qD = np.array([q11D, q21D])
	ksiD = np.linalg.inv(A) @ B @ qD
	xD = ksiD[0]
	yD = ksiD[1]
	return xD, yD


def ikm(q11, q12, q21, q22, xD, yD): # correct
	A, B = compute_A_B(q11, q12, q21, q22)
	ksiD = np.array([xD, yD])
	qD = np.linalg.inv(B) @ A @ ksiD
	q11D = qD[0]
	q21D = qD[1]
	return q11D, q21D


def dkm_passive(q11, q12, q21, q22, q11D, q21D, xD, yD):
	qD = np.array([q11D, q21D])
	ksiD = np.array([xD, yD])
	v12 = get_v12(get_u12(q11, q12))
	v22 = get_v22(get_u22(q21, q22))
	v11 = get_v11(get_u11(q11))
	v21 = get_v21(get_u21(q21))

	matrix_helper = np.array([
		[l*np.dot(v12, v11) + l, 0], \
		[0, l* np.dot(v22, v21) + l]
		])
	
	matrix_helper_ksiD = np.array([v12.T, v22.T])
	qPD = np.dot(matrix_helper_ksiD, ksiD) - matrix_helper @ qD
	qPD = qPD/l
	q12D = qPD[0]
	q22D = qPD[1]
	return q12D, q22D

def get_d_matrix(q11, q21, q12, q22, q11D, q12D, q21D, q22D, l):
	u11 = get_u11(q11)
	u21 = get_u21(q21)
	u12 = get_u12(q11, q12)
	u22 = get_u22(q21, q22)
	v12 = get_v12(u12)
	v22 = get_v22(u22)
	return np.array([
		-l * q11D**2 * np.dot(u12.T, u11) - l*(q11D + q12D)**2, -l * q21D**2 * np.dot(u22.T, u21) - l*(q21D + q22D)**2])

def dkm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD):
	A, B = compute_A_B(q11, q12, q21, q22)
	d_matrix = get_d_matrix(q11, q21, q12, q22, q11D, q12D, q21D, q22D, l)
	qDD = np.array([q11DD, q21DD])
	ksiDD = np.linalg.inv(A) @ (B @ qDD + d_matrix) 

	xDD = ksiDD[0]
	yDD = ksiDD[1]
	return xDD, yDD


def ikm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, xDD, yDD):
	A, B = compute_A_B(q11, q12, q21, q22)
	d_matrix = get_d_matrix(q11, q21, q12, q22, q11D, q12D, q21D, q22D, l)
	
	ksiDD = np.array([xDD, yDD])
	qDD = np.linalg.inv(B) @ ((A @ ksiDD) - d_matrix) 

	q11DD = qDD[0]
	q21DD = qDD[1] 
	return q11DD, q21DD

def dkm2_passive(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD, xDD, yDD): 
	qDD = np.array([q11DD, q21DD])
	ksiDD = np.array([xDD, yDD])
	
	u11 = get_u11(q11)
	u12 = get_u12(q11, q12)
	u21 = get_u21(q21)
	v11 = get_v11(u11)
	v12 = get_v12(u12)
	v22 = get_v22(get_u22(q21, q22))
	v21 = get_v21(get_u21(q21))
	
	matrix_helper_ksiDD = np.vstack((v12.T, v22.T))
	matrix_helper_qDD = np.vstack((
			np.array([l*np.dot(v12, v11) + l, 0]), \
			np.array([0, l* np.dot(v22, v21) + l])
			))
	
	vector_residual = np.array([
		-l * q11D**2 * np.dot(v12, u11), \
	    -l * q21D**2 * np.dot(v22, u21)])

	qPDD = np.dot(matrix_helper_ksiDD, ksiDD) - matrix_helper_qDD @ qDD - vector_residual
	qPDD = qPDD/l

	q12DD = qPDD[0]
	q22DD = qPDD[1]
	return q12DD, q22DD


def dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D):
	Z = np.array([[ZZ1R, 0], [0, ZZ2R]])

	A, B = compute_A_B(q11, q12, q21, q22)
	d_matrix = get_d_matrix(q11, q21, q12, q22, q11D, q12D, q21D, q22D, l)
	J = np.dot(np.linalg.inv(A), B)
	M = Z + mR*J.T @ np.linalg.inv(A) @ B
	c = mR * J.T @ np.linalg.inv(A) @ d_matrix
	return M, c
