from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    B = tvm.te.placeholder(shape, dtype=dtype, name="B")
    C = tvm.te.compute(shape, lambda *i: A(*i) * B(*i))

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = tvm.te.compute(A.shape, lambda *i: A(*i) + const_k)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = tvm.te.compute(shape, lambda *i: A(*i) * const_k)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = tvm.te.compute(shape, lambda *i: tvm.te.max(A(*i), 0))
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    gradient = tvm.te.placeholder(shape, dtype=dtype, name="G")
    C = tvm.te.compute(
        shape, lambda *i: tvm.tir.Select(A(*i) > 0, gradient(*i), float(0)))
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, gradient, C], tgt,
                  target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""

    # Get the output C shape
    m, inner_k = (shapeA[1], shapeA[0]) if transposeA else (
        shapeA[0], shapeA[1])
    n = shapeB[0] if transposeB else shapeB[1]

    shapeC = (m, n)
    k = tvm.te.reduce_axis((0, inner_k), name="k")
    A = tvm.te.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.te.placeholder(shapeB, dtype=dtype, name="B")

    if transposeA == False and transposeB == False:
        bs = 32 #block size in C matrix
        ks = 4 #K dim block size 

        #To make B access a sequential pattern while compute a C matrix tile
        packedB = tvm.te.compute(((shapeB[1]+bs-1) / bs, shapeB[0], bs), lambda x, y, z: B[y, x * bs + z], name='packedB')
        C = tvm.te.compute(shapeC, lambda x, y: tvm.te.sum(A[x][k] * packedB[y // bs][k][tvm.tir.indexmod(y, bs)], axis=k), name = 'C')

        # C = tvm.te.compute(shapeC, lambda x, y: tvm.te.sum(A[x, k] * B[k, y], axis=k), name = 'C')


        s = tvm.te.create_schedule(C.op)

        # Allocate write cache
        CCache = s.cache_write(C, 'global')

        # Blocking by loop tiling
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bs, bs)

        # Write cache is computed at yo
        s[CCache].compute_at(s[C], yo)
        # New inner axes
        xc, yc = s[CCache].op.axis

        k, = s[CCache].op.reduce_axis
        ko, ki = s[CCache].split(k, factor=ks)

        s[CCache].reorder(ko, xc, ki, yc)
        s[CCache].unroll(ki)
        s[CCache].vectorize(yc)
        # parallel
        s[C].parallel(xo)

        # Vercorization for packed B and parallel for it
        xb, yb, zb = s[packedB].op.axis
        s[packedB].vectorize(zb)
        s[packedB].parallel(xb)

        f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
        #print(tvm.lower(s, [A, B, C], name=func_name, simple_mode=True))
        return f

    if transposeA == False and transposeB == True:
        bs =64
        # ks = int(64/4 * 2)
        ks = 8

        # not use pack
        C = tvm.te.compute(shapeC, lambda i, j: tvm.te.sum(A[i][k] * B[j][k], axis=k), name="C")
        s = tvm.te.create_schedule(C.op)

        # Allocate write cache
        CCache = s.cache_write(C, 'global')

        # Blocking by loop tiling
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bs, bs)

        # Write cache is computed at yo
        s[CCache].compute_at(s[C], yo)
        # New inner axes
        xc, yc = s[CCache].op.axis

        k, = s[CCache].op.reduce_axis
        ko, ki = s[CCache].split(k, factor=ks)

        s[CCache].reorder(ko, xc, yc, ki)
        s[CCache].unroll(ki)
        s[CCache].unroll(yc)
        #s[CCache].unroll(xc)

        #s[CCache].vectorize(ki)
        # parallel
        s[C].parallel(xo)

        # Vectorize for write back to output C
        s[C].vectorize(yi)

        f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
        # print(tvm.lower(s, [A, B, C], name=func_name, simple_mode=True))
        return f

    if transposeA == True and transposeB == False:
        # Add all optimization for A & B
        bs = 32 
        ks = 4

        #To make A access a sequential pattern while compute a C matrix tile
        packedA = tvm.te.compute(((shapeA[1]+bs-1) / bs, shapeA[0], bs), lambda xa, ya, za: A[ya, xa* bs + za], name='packedA')

        #To make B access a sequential pattern while compute a C matrix tile
        packedB = tvm.te.compute(((shapeB[1]+bs-1)/ bs, shapeB[0], bs), lambda xb, yb, zb: B[yb, xb * bs + zb], name='packedB')

        C = tvm.te.compute(shapeC, lambda i, j: tvm.te.sum(A[k][i] * B[k][j], axis=k), name="C")
        C = tvm.te.compute(shapeC, lambda i, j: tvm.te.sum(packedA[ i // bs ][k][tvm.tir.indexmod(i, bs)] * packedB[j//bs][k][tvm.tir.indexmod(j, bs)], axis=k), name="C")

        # C = tvm.te.compute(shapeC, lambda i, j: tvm.te.sum(A[k][i] * B[k][j], axis=k), name="C") # naive version
        s = tvm.te.create_schedule(C.op)

        #########
        # Allocate write cache
        CCache = s.cache_write(C, 'global')
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bs, bs)
        # New inner axes
        xc, yc = s[CCache].op.axis
        # Blocking by loop tiling
        # Write cache is computed at yo
        s[CCache].compute_at(s[C], yo)

        k, = s[CCache].op.reduce_axis
        ko, ki = s[CCache].split(k, factor=ks)


        s[CCache].reorder(ko, ki, xc, yc)
        # s[CCache].unroll(xc)
        s[CCache].vectorize(yc)


        # # For packed array
        xa, ya, za = s[packedA].op.axis
        xb, yb, zb = s[packedB].op.axis
        s[packedA].reorder(ya,xa,za)
        s[packedB].reorder(yb,xb,zb)

        s[packedA].vectorize(za)
        s[packedB].vectorize(zb)
        s[packedA].parallel(xa)
        s[packedB].parallel(xb)

        # # Vectorize for write back to output C
        s[C].vectorize(yi)
        s[C].parallel(xo)
        # s[C].reorder(ko, ki, xi, yi)
        # s[C].vectorize(yi)
        f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
        # print(tvm.lower(s, [A, B, C], name=func_name, simple_mode=True))
        return f

    if transposeA == True and transposeB == True:
        # Add pack for A
        # skip pack for B
        bs = 32
        ks = 8
        
        packedA = tvm.te.compute(((shapeA[1]+bs-1) / bs, shapeA[0], bs), lambda xa, ya, za: A[ya, xa* bs + za], name='packedA')

        # C = tvm.te.compute(shapeC, lambda i, j: tvm.te.sum(A[k][i] * B[j][k], axis=k), name="C")
        C = tvm.te.compute(shapeC, lambda i, j: tvm.te.sum(packedA[ i // bs ][k][tvm.tir.indexmod(i, bs)] * B[j][k], axis=k), name="C")
        s = tvm.te.create_schedule(C.op)
        ################################
        
        # Allocate write cache
        CCache = s.cache_write(C, 'global')
        xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bs, bs)

        # New inner axes
        xc, yc = s[CCache].op.axis
        s[CCache].compute_at(s[C], yo)

        k, = s[CCache].op.reduce_axis
        ko, ki = s[CCache].split(k, factor=ks)
        
        s[CCache].reorder(ko, yc, ki, xc)
        s[CCache].unroll(ki)
        s[CCache].vectorize(xc)

        s[C].parallel(xo)


        xa, ya, za = s[packedA].op.axis
        s[packedA].vectorize(za)
        s[packedA].parallel(xa)
        f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
        # print(tvm.lower(s, [A, B, C], name=func_name, simple_mode=True))
        return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    out_H = H - R + 1
    out_W = W - S + 1
    shapeC = (N, M, out_H, out_W)

    c = tvm.te.reduce_axis((0, C), name="input_C")  # Input Channel
    r = tvm.te.reduce_axis((0, R), name="r")
    s = tvm.te.reduce_axis((0, S), name="s")

    X = tvm.te.placeholder(shapeX, dtype=dtype, name="X")
    F = tvm.te.placeholder(shapeF, dtype=dtype, name="F")
    C = tvm.te.compute(shapeC, lambda ni, mi, hi, wi: tvm.te.sum(
        X[ni][c][hi+r][wi+s]*F[mi][c][r][s], [c, r, s]))
    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [X, F, C], tgt, target_host=tgt_host, name=func_name)
    # f = tvm.lower(s, [X, F, C], name=func_name, simple_mode=True)
    # print(f)
    return f


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")

    k = tvm.te.reduce_axis((0, shape[1]), name="k")
    maxtrix_row_max = tvm.te.compute(
        (shape[0],), lambda i: tvm.te.max(A[i][k], k), name="row_max")
    Ex = tvm.te.compute(shape, lambda i, j: tvm.te.exp(
        A[i][j] - maxtrix_row_max[i]), name="exp_element")

    h = tvm.te.reduce_axis((0, shape[1]), name="h")
    Ex_sum = tvm.te.compute(
        (shape[0],), lambda i: tvm.te.sum(Ex[i][h], h), name="row_sum")
    C = tvm.te.compute(shape, lambda i, j: Ex[i][j]/Ex_sum[i], name="soft_max")

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    #print(tvm.lower(s, [A, C],name=func_name, simple_mode=True))
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""

    # softmax
    y = tvm.te.placeholder(shape, dtype=dtype, name="y")  # input y
    maxtrix_row_max = topi.max(y, axis=1, keepdims=False)
    Ex = tvm.te.compute(shape, lambda i, j: tvm.te.exp(y[i][j] - maxtrix_row_max[i]), name="exp_element")
    Ex_sum = topi.sum(Ex, axis=1, keepdims=False)

    soft_max = tvm.te.compute(shape, lambda i, j: Ex[i][j]/Ex_sum[i], name="soft_max")

    # cross_entropy
    y_real = tvm.te.placeholder(shape, dtype=dtype, name="y_real")
    j = tvm.te.reduce_axis((0, shape[1]), name="j")
    loss = tvm.te.compute((shape[0],), lambda i: tvm.te.sum(y_real[i][j] * tvm.te.log(soft_max[i][j]), j), name="loss")

    sum_loss = topi.sum(loss, axis = 0, keepdims=True)
    mean_loss = tvm.te.compute((1,), lambda *i: -1 * sum_loss(*i)/shape[0], "mean_loss")

    s = tvm.te.create_schedule(mean_loss.op)
    f = tvm.build(s, [y, y_real, mean_loss], tgt,
                  target_host=tgt_host, name=func_name)
    # print(tvm.lower(s, [y, y_real, mean_loss],name=func_name, simple_mode=True))
    return f


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.te.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.te.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.te.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.te.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.te.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.te.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
