import numpy
from numpy.polynomial import Polynomial
import ampal
import copy
from ampal.geometry import angle_between_vectors, dihedral, Quaternion, find_foot, unit_vector
from scipy.signal import convolve2d

def sign(a, b):
    return abs(a) if b >= 0 else -abs(a)

def poly_mul1(u1, u2, p1_in, p2_in):
    # Ensure inputs are treated as polynomials of the correct degree
    res = Polynomial(u1[:p1_in+1]) * Polynomial(u2[:p2_in+1])
    # The degree should be p1_in + p2_in.
    # The length of the coefficient array should be degree + 1
    degree = p1_in + p2_in
    output = numpy.zeros(17)
    output[:len(res.coef)] = res.coef
    return output, degree

def poly_sub1(u1, u2, p1_in, p2_in):
    res = Polynomial(u1[:p1_in+1]) - Polynomial(u2[:p2_in+1])
    degree = max(p1_in, p2_in)
    output = numpy.zeros(17)
    output[:len(res.coef)] = res.coef
    return output, degree

def poly_mul_sub1(u1, u2, u3, u4, p1_in, p2_in, p3_in, p4_in):
    d1, pd1 = poly_mul1(u1, u2, p1_in, p2_in)
    d2, pd2 = poly_mul1(u3, u4, p3_in, p4_in)
    return poly_sub1(d1, d2, pd1, pd2)

def poly_mul2(u1, u2, p1_in, p2_in):
    u3 = numpy.zeros([5, 5])
    p3 = [x + y for x, y in zip(p1_in, p2_in)]
    slice1 = u1[:p1_in[1]+1, :p1_in[0]+1]
    slice2 = u2[:p2_in[1]+1, :p2_in[0]+1]
    u3[:p3[1]+1, :p3[0]+1] = convolve2d(slice1, slice2)
    return u3, p3

def poly_sub2(u1, u2, p1_in, p2_in):
    p3 = [max(p1_in[0], p2_in[0]), max(p1_in[1], p2_in[1])]
    u3 = numpy.zeros([5, 5])
    u3[:p3[1]+1, :p3[0]+1] = u1[:p3[1]+1, :p3[0]+1] - u2[:p3[1]+1, :p3[0]+1]
    return u3, p3

def poly_mul_sub2(u1, u2, u3, u4, p1_in, p2_in, p3_in, p4_in):
    d1, pd1 = poly_mul2(u1, u2, p1_in, p2_in)
    d2, pd2 = poly_mul2(u3, u4, p3_in, p4_in)
    return poly_sub2(d1, d2, pd1, pd2)

def calc_t1(t0, t2, C0, C1, C2):
    t0_2 = t0 ** 2
    t2_2 = t2 ** 2
    U11 = C0[0,0] + C0[0,1] * t0 + C0[0,2] * t0_2
    U12 = C1[0,0] + C1[0,1] * t0 + C1[0,2] * t0_2
    U13 = C2[0,0] + C2[0,1] * t0 + C2[0,2] * t0_2
    U31 = C0[1,0] + C0[1,1] * t2 + C0[1,2] * t2_2
    U32 = C1[1,0] + C1[1,1] * t2 + C1[1,2] * t2_2
    U33 = C2[1,0] + C2[1,1] * t2 + C2[1,2] * t2_2
    return (U31 * U13 - U11 * U33) / (U12 * U33 - U13 * U32)

def calc_t2(t0, Q, R):
    t0_2 = t0 * t0
    t0_3 = t0_2 * t0
    t0_4 = t0_3 * t0
    A = [Q[i,0] + Q[i,1] * t0 + Q[i,2] * t0_2 + Q[i,3] * t0_3 + Q[i,4] * t0_4 for i in range(5)]
    B = [R[i,0] + R[i,1] * t0 + R[i,2] * t0_2 for i in range(3)]
    B2_2 = B[2] * B[2]
    B2_3 = B2_2 * B[2]
    K0 = A[2] * B[2] - A[4] * B[0]
    K1 = A[3] * B[2] - A[4] * B[1]
    K2 = A[1] * B2_2 - K1 * B[0]
    K3 = K0 * B[2] - K1 * B[1]
    return (K3 * B[0] - A[0] * B2_3) / (K2 * B[2] - K3 * B[1])

def new_input_angles(input_prot, index1, index2, index3, include_original=True):
    Natoms = [input_prot[x]['N'].array for x in [index1, index2, index3]]
    CAatoms = [input_prot[x]['CA'].array for x in [index1, index2, index3]]
    Catoms = [input_prot[x]['C'].array for x in [index1, index2, index3]]
    
    minus_idx = [x%3 for x in range(-1, 2)]
    plus_idx = [x%3 for x in range(1, 4)]
    
    alphas = numpy.array([angle_between_vectors((CAatoms[y]-CAatoms[x]), (CAatoms[x]-CAatoms[z]), radians=True)
                         for x, y, z in zip(range(3), plus_idx, minus_idx)])
    alphas[1] = numpy.pi - alphas[1]
    
    etas = numpy.array([angle_between_vectors((Catoms[x]-CAatoms[x]), (CAatoms[y]-CAatoms[x]), radians=True)
                       for x, y in zip(range(3), plus_idx)])
    etas[2] = numpy.pi - etas[2]

    xis = numpy.array([angle_between_vectors((CAatoms[y]-CAatoms[x]),(Natoms[x]-CAatoms[x]), radians=True)
                       for x, y in zip(range(3), minus_idx)])
    xis[0] = numpy.pi - xis[0]

    deltas = numpy.array([dihedral(Catoms[x], CAatoms[x], Catoms[y], Natoms[y], radians=True)
                          for x, y in zip(minus_idx, range(3))])
    deltas[1] = numpy.pi - deltas[1]
    deltas[2] = numpy.pi - deltas[2]
    
    zhats = numpy.array([(CAatoms[y] - CAatoms[x]) / 
                           numpy.sqrt(sum((CAatoms[y] - CAatoms[x])**2)) for  x, y in zip(range(3), plus_idx)])
    yhat = numpy.cross(zhats[2], zhats[0])/numpy.sqrt(sum(numpy.cross(zhats[2], zhats[0])**2))
    xhats = numpy.array([numpy.cross(yhat, x) for x in zhats])
    
    feet = [find_foot(Catoms[x], CAatoms[x], Catoms[y]) for x, y in zip(range(3), plus_idx)]
    
    ref_tau1 = numpy.pi - dihedral(Catoms[2], CAatoms[0], CAatoms[2], (CAatoms[2]+yhat), radians = True)
    ref_tau2 = dihedral((CAatoms[0]-yhat), CAatoms[0], CAatoms[1], Catoms[0], radians = True)
    ref_tau3 = dihedral((CAatoms[1]+yhat), CAatoms[1], CAatoms[2], Catoms[1], radians = True)
    
    ref_taus = [ref_tau1, ref_tau2, ref_tau3]
    r_tmps = [unit_vector(Natoms[i]-CAatoms[i]) - xhats[i]*numpy.cos(xis[i])/numpy.sin(xis[i]) for i in range(3)]
    init_angles = [angle_between_vectors(zhats[i], r_tmps[i], radians=True) for i in range(3)]
    sig_inits = [sign(init_angles[i], numpy.dot(r_tmps[i], yhat)) for i in range(3)]
    tau_inits = [sig_inits[2]-deltas[0], sig_inits[0]-deltas[1], sig_inits[1]-deltas[2]]
    thetas = numpy.array([angle_between_vectors((CAatoms[x] - Natoms[x]), (CAatoms[x] - Catoms[x]), radians=True) for x in range(3)])
    
    B0 = numpy.zeros(3); B1 = numpy.zeros(3); B2 = numpy.zeros(3); B3 = numpy.zeros(3); B4 = numpy.zeros(3); B5 = numpy.zeros(3); B6 = numpy.zeros(3); B7 = numpy.zeros(3); B8 = numpy.zeros(3)
    C0 = numpy.zeros([3, 3]); C1 = numpy.zeros([3, 3]); C2 = numpy.zeros([3, 3])
    u11 = numpy.zeros([5, 5]); u12 = numpy.zeros([5, 5]); u13 = numpy.zeros([5, 5]); u31 = numpy.zeros([5, 5]); u32 = numpy.zeros([5, 5]); u33 = numpy.zeros([5, 5])

    for i in range(3):
        A0 = numpy.cos(alphas[i]) * numpy.cos(xis[i]) * numpy.cos(etas[i]) - numpy.cos(thetas[i])
        A1 = -numpy.sin(alphas[i]) * numpy.cos(xis[i]) * numpy.sin(etas[i])
        A2 = numpy.sin(alphas[i]) * numpy.sin(xis[i]) * numpy.cos(etas[i])
        A3 = numpy.sin(xis[i]) * numpy.sin(etas[i])
        A4 = A3 * numpy.cos(alphas[i])
        A21 = A2 * numpy.cos(deltas[i]); A22 = A2 * numpy.sin(deltas[i])
        A31 = A3 * numpy.cos(deltas[i]); A32 = A3 * numpy.sin(deltas[i])
        A41 = A4 * numpy.cos(deltas[i]); A42 = A4 * numpy.sin(deltas[i])
        B0[i] = A0 + A22 + A31
        B1[i] = 2 * (A1 + A42)
        B2[i] = 2 * (A32 - A21)
        B3[i] = -4 * A41
        B4[i] = A0 + A22 - A31
        B5[i] = A0 - A22 - A31
        B6[i] = -2 * (A21 + A32)
        B7[i] = 2 * (A1 - A42)
        B8[i] = A0 - A22 + A31

    C0[0,0] = B0[0]; C0[0,1] = B2[0]; C0[0,2] = B5[0]
    C1[0,0] = B1[0]; C1[0,1] = B3[0]; C1[0,2] = B7[0]
    C2[0,0] = B4[0]; C2[0,1] = B6[0]; C2[0,2] = B8[0]

    for i in range(1, 3):
        C0[i,0] = B0[i]; C0[i,1] = B1[i]; C0[i,2] = B4[i]
        C1[i,0] = B2[i]; C1[i,1] = B3[i]; C1[i,2] = B6[i]
        C2[i,0] = B5[i]; C2[i,1] = B7[i]; C2[i,2] = B8[i]

    for i in range(3):
        u11[0,i] = C0[0,i]; u12[0,i] = C1[0,i]; u13[0,i] = C2[0,i]
        u31[i,0] = C0[1,i]; u32[i,0] = C1[1,i]; u33[i,0] = C2[1,i]

    p1 = [2, 0]; p3 = [0, 2]
    um1, p_um1 = poly_mul_sub2(u32, u32, u31, u33, p3, p3, p3, p3)
    um2, p_um2 = poly_mul_sub2(u12, u32, u11, u33, p1, p3, p1, p3)
    um3, p_um3 = poly_mul_sub2(u12, u33, u13, u32, p1, p3, p1, p3)
    um4, p_um4 = poly_mul_sub2(u11, u33, u31, u13, p1, p3, p3, p1)
    um5, p_um5 = poly_mul_sub2(u13, um1, u33, um2, [1,1], [2,2], [0,2], [2,2])
    um6, p_um6 = poly_mul_sub2(u13, um4, u12, um3, [1,1], [2,2], [1,1], [2,2])
    q_tmp, p_Q = poly_mul_sub2(u11, um5, u31, um6, [1,1], [2,2], [0,2], [2,2])
    Q = q_tmp[:]
    R = numpy.zeros([3, 17])
    for i in range(3):
        R[0,i] = C0[2,i]; R[1,i] = C1[2,i]; R[2,i] = C2[2,i]
    p2 = 2; p4 = 4
    f1, _ = poly_mul_sub1(R[1], R[1], R[0], R[2], p2, p2, p2, p2)
    f2, _ = poly_mul1(R[1], R[2], p2, p2)
    f3, _ = poly_mul_sub1(R[1], f1, R[0], f2, p2, p2, p2, p2)
    f4, _ = poly_mul1(R[2], f1, p2, p2)
    f5, _ = poly_mul_sub1(R[1], f3, R[0], f4, p2, p2, p2, p2)
    f6, _ = poly_mul_sub1(Q[1], R[1], Q[0], R[2], p4, p2, p4, p2)
    f7, _ = poly_mul_sub1(Q[2], f1, R[2], f6, p4, p2, p2, p4)
    f8, _ = poly_mul_sub1(Q[3], f3, R[2], f7, p4, p2, p2, p4)
    f9, _ = poly_mul_sub1(Q[4], f5, R[2], f8, p4, p2, p2, p4)
    f10, _ = poly_mul_sub1(Q[3], R[1], Q[4], R[0], p4, p2, p4, p2)
    f11, _ = poly_mul_sub1(Q[2], f1, R[0], f10, p4, p2, p2, p4)
    f12, _ = poly_mul_sub1(Q[1], f3, R[0], f11, p4, p2, p2, p4)
    f13, _ = poly_mul_sub1(Q[2], R[1], Q[1], R[2], p4, p2, p4, p2)
    f14, _ = poly_mul_sub1(Q[3], f1, R[2], f13, p4, p2, p2, p4)
    f15, _ = poly_mul_sub1(Q[3], R[1], Q[2], R[2], p4, p2, p4, p2)
    f16, _ = poly_mul_sub1(Q[4], f1, R[2], f15, p4, p2, p2, p4)
    f17, _ = poly_mul_sub1(Q[1], f14, Q[0], f16, p4, p4, p4, p4)
    f18, _ = poly_mul_sub1(Q[2], R[2], Q[3], R[1], p4, p2, p4, p2)
    f19, _ = poly_mul_sub1(Q[1], R[2], Q[3], R[0], p4, p2, p4, p2)
    f20, _ = poly_mul_sub1(Q[3], f19, Q[2], f18, p4, p4, p4, p4)
    f21, _ = poly_mul_sub1(Q[1], R[1], Q[2], R[0], p4, p2, p4, p2)
    f22, _ = poly_mul1(Q[4], f21, p4, p4)
    f23, _ = poly_sub1(f20, f22, p4, p4)
    f24, _ = poly_mul1(R[0], f23, p2, p4)
    f25, _ = poly_sub1(f17, f24, p4, p4)
    f26, _ = poly_mul_sub1(Q[4], f12, R[2], f25, p4, p4, p2, p4)
    poly_coeff, _ = poly_mul_sub1(Q[0], f9, R[0], f26, p4, p4, p2, p4)
    
    if poly_coeff[16] < 0.0:
        poly_coeff *= -1
    
    all_roots = numpy.roots(numpy.flip(poly_coeff))
    roots = all_roots[numpy.isreal(all_roots)].real
    net_rotation_list2 = []
    for i in range(len(roots)):
        half_tan = numpy.zeros(3)
        half_tan[2] = roots[i]
        half_tan[1] = calc_t2(half_tan[2], Q, R)
        half_tan[0] = calc_t1(half_tan[2], half_tan[1], C0, C1, C2)
        cos_tau, sin_tau = numpy.zeros(4), numpy.zeros(4)
        for j in range(1, 4):
            ht = half_tan[j - 1]
            tmp = 1.0 + ht ** 2
            cos_tau[j] = (1.0 - ht ** 2) / tmp
            sin_tau[j] = 2.0 * ht / tmp
        cos_tau[0], sin_tau[0] = cos_tau[3], sin_tau[3]
        taus = [numpy.arctan2(sin_tau[i], cos_tau[i]) for i in range(3)]
        net_rotations2 = [ref_taus[0] - taus[0], taus[1] - ref_taus[1], taus[2] - ref_taus[2]]
        net_rotation_list2.append([net_rotations2[1], net_rotations2[2], net_rotations2[0]])        
        
    output_structures = ampal.AmpalContainer()
    idx_list = [(index1, index2), (index2, index3), (index1, index3)]
    axes = [input_prot[x[1]].atoms['CA']._vector - input_prot[x[0]].atoms['CA']._vector for x in idx_list]
    for i in range(len(net_rotation_list2)):
        output_struc = copy.deepcopy(input_prot.backbone)
        for j in range(3):
            quat = Quaternion.angle_and_axis(angle=net_rotation_list2[i][j], axis=axes[j], radians=True)
            atoms = [output_struc[idx_list[j][0]][_] for _ in list(output_struc[idx_list[j][0]].atoms.keys())[2:]]
            for k in range(idx_list[j][0]+1, idx_list[j][1]): atoms += [output_struc[k][_] for _ in list(output_struc[k].atoms.keys())]
            atoms.append(output_struc[idx_list[j][1]].atoms['N'])
            for atom in atoms:
                atom._vector = quat.rotate_vector(v=atom._vector, point= output_struc[idx_list[j][0]].atoms['CA']._vector)
        output_structures.append(output_struc)
    return output_structures
