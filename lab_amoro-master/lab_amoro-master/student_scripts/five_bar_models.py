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
def A22H(q11, q21):
	A22H = 1/2*np.array([-l * cos(q21) - d + l*cos(q11), -l*sin(q21) + l * sin(q11)])
	return A22H

def A21A22(q21):
	A21A22 = np.array([l * sin(q21), l* cos(q21)])
	return A21A22

def dgm(q11, q21, assembly_mode):
	A22H = A22H(q11, q21)
	dA22H = np.linalg.norm(A22H)
	h = sqrt(l**2 - dA22H**2)
	HA13 = assembly_mode * (h / dA22H) * np.array([[0, -1], [1, 0]]) @ A22H
	OA21 = np.array([d/2., 0.])
	A21A22 = A21A22(q21)
	OA13 = OA21 + A21A22 + A22H + HA13
	
	x = OA13[0]
	y = OA13[1]
	return x, y

def A11A13(x, y, d):
	A11A13 = np.array([d/2 + x, y])
	return A11A13

def igm(x, y, gamma1, gamma2):
	A11A13 = A11A13(x, y, d)
	A11M1 = 1/2 * A11A13
	c = np.linalg.norm(A11M1)
	b = sqrt(l**2 - c**2)
	M1A12 =  gamma1 * b/c * np.array([[0, -1], [1, 0]])
	A11A12 = A11M1 + M1A12 
	q11 = atan2(A11A12[0], A11A12[1])
	q21 = atan2(A21A22[0], A21A22[1])
	return q11, q21


def dgm_passive(q11, q21, assembly_mode):
    q12 = atan2(y/l - sin(q11), x/l + d/(2 * l) - cos(q11)) - q11
    q22 = atan2(y/l - sin(q21), x/l - d/(2 * l) - cos(q21)) - q21
    return q12, q22

def u11(A11A12):
	return A11A12

def u12(A12A13):
	return A12A13

def u21(A21A22):
	return A21A22

def u22(A22A13):
	return A22A13

def vector_90_rotation(vector):
	rotated_vector = np.zeros(2)
	rotated_vector[0] = vector[1]
	rotated_vector[1] = vector[0]*-1
	return rotated_vector

def v11(u11):
	return vector_90_rotation(u11)

def v12(u12):
	return vector_90_rotation(u12)

def v21(u21):
	return vector_90_rotation(u21)

def v22(u22):
	return vector_90_rotation(u22)

# You can create intermediate functions to avoid redundant code
def compute_A_B(q11, q12, q21, q22):
    A = np.array([u12.T, u22.T])
    B = np.array([[l*np.dot(u12, v11), 0], [0, l*np.dot(u22, v21)]])
    return A, B

def dkm(q11, q12, q21, q22, q11D, q21D):
	A, B = compute_A_B(q11, q12, q21, q22)
	qD = np.array([q11D, q21D])
	ksi = np.matmul(np.matmul(np.linalg.inv(A), B), qD)
	xD = ksi[0]
	yD = ksi[1]
	return xD, yD


def ikm(q11, q12, q21, q22, xD, yD):
	A, B = compute_A_B(q11, q12, q21, q22)
	ksi = np.array([xD, yD])
	qD = np.matmul(np.matmul(np.linalg.inv(B), A), ksi)
	q11D = qD[0]
	q21D = qD[1]
	return q11D, q21D


def dkm_passive(q11, q12, q21, q22, q11D, q21D, xD, yD):
    qD = np.array([q11D, q21D])
    ksi = np.array([xD, yD])
    qP = np.dot(np.array([v12.T, v22.T]), ksi) - np.matmul(np.array([[l*np.dot(v12, v11) + l, 0], [0, l* np.dot(v22, v21) + l]]), qD)
    qP = qP/l
    q12D = qP[0]
    q22D = qP[1]
    return q12D, q22D

# i need to change previous ksi to ksiD where it applies
# also for qP

def dkm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD):
	A, B = compute_A_B(q11, q12, q21, q22)
	qD = np.array([q11D, q21D])
	qDD = np.array([q11DD, q21DD])
	ksiDD = np.matmul(np.linalg.inv(A), np.matmul(B, qDD) + d)

	xDD = ksiDD[0]
	yDD = ksiDD[1]
	return xDD, yDD

def ikm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, xDD, yDD):
	A, B = compute_A_B(q11, q12, q21, q22)
	qD = np.array([q11D, q21D])
	ksiDD = np.array([xDD, yDD])
	qDD = np.matmul(np.linalg.inv(B), np.matmul(A, ksiDD) - d)

	q11DD = qDD[0]
	q21DD = qDD[1]
	return q11DD, q21DD

def dkm2_passive(q11, q12, q21, q22, q11D, q12D, q21D, q22D, q11DD, q21DD, xDD, yDD):
	qD = np.array([q11D, q21D])
	qDD = np.array([q11DD, q21DD])
	ksiDD = np.array([xDD, yDD])
	qP = np.dot(np.array([v12.T, v22.T]), ksiDD) - np.matmul(np.array([[l*np.dot(v12, v11) + l, 0], [0, l* np.dot(v22, v21) + l]]), qDD) - np.array([-l * q11d**2 * np.dot(v12, u11), -l * q21d**2 * np.dot(v22, u21)])
	qP = qP/l

	q12DD = qP[0]
	q22DD = qP[1]
	return q12DD, q22DD


def dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D):
	Z = np.array([[ZZ1R, 0], [0, ZZ2R]])

	A, B = compute_A_B(q11, q12, q21, q22)
	J = np.dot(np.linalg.inv(A), B)
	M = Z + m*J.T @ np.linalg.inv(A) @ B
	c = m * J.T @ np.linalg.inv(A) * d
	return M, c
