import numpy as np
import math
from numpy import linalg as LA
import sys
import time


def ComputeLipschitz(alpha, beta, eta, p, q, N):
    L1 = p * (alpha**(p - 2)) / beta**p
    L2 = p / (2 * alpha**2) * max(1, (N * alpha ** p / beta**p)**2)
    L3 = (q - 1) / eta**2
    return L1 + L2 + L3


def Lpsmooth(x, alpha, p):
    """
    This function computes the smooth Lp norm of the vector x
    """
    res = np.sum((x**2 + alpha**2) ** (p / 2) - alpha**p)
    res = res**(1 / p)
    return res


def Lqsmooth(x, mu, q):
    """
    This function computes the smooth Lq norm of the vector x
    """
    res = mu**q + np.sum(np.abs(x)**q)
    res = res**(1 / q)
    return res


def Fcost(x, alpha, beta, mu, p, q):
    lp = (np.sum((x**2 + alpha**2)**(p / 2)) - alpha**p)**(1 / q)
    lq = (mu**q + np.sum(np.abs(x)**q))**(1 / q)
    fcost = math.log(((lp**p + beta**p)**(1 / q)) / lq)
    return fcost


def condlplq(x, alpha, beta, eta, p, q, ro):
    """
    This function computes the metric matrix for the variable metric Forward-Backward algorithm
    """
    lp = Lpsmooth(x, alpha, p)
    Xpq = (q - 1) / ((eta**q + ro**q)**(2 / q))
    A = Xpq + (1 / (lp ** p + beta**p)) * ((x**2 + alpha**2)**(p / 2 - 1))
    return A


def gradlplq(x, alpha, beta, mu, p, q):
    """
    his function computes the gradient of smooth lp over lq function
    """
    lp = Lpsmooth(x, alpha, p)
    lq = Lqsmooth(x, mu, q)
    grad1 = x * ((x**2 + alpha**2)**(p / 2 - 1)) / (lp ** p + beta**p)
    grad2 = np.sign(x) * (np.abs(x)**(q - 1)) / (lq ** q)
    return grad1 - grad2


def norm2(K, N, nbiter=50):
    b = np.random.rand(N, 1)
    K_transpose = K.transpose()
    i = 0
    while i < nbiter:
        tmp = np.dot(K_transpose, np.dot(K, b))
        tmp_norm = LA.norm(tmp)
        b = tmp / tmp_norm
        i += 1
    return LA.norm(np.dot(K, b))


def proxB(B, x, xhat, teta):
    """
    This function computes the proximity operators of f(x) = (teta/2) * ||y-x||_B^2
    """
    p = (x + teta * (B * xhat)) / (1 + teta * B)
    p[p < 0] = 0
    return p


def proxl1(x, w):
    """
    proximity operator of l1 norm: Thresholding y = max(abs(x)-w,0).*sign(x);
    """
    return np.sign(x) * np.maximum(np.abs(x) - w, 0.)


def proxl2(x, y, eta):
    """
    projection onto the l2 ball
    """
    t = x - y
    s = t * min(eta / LA.norm(t), 1)
    return x + s - t


def proxPPXAplus(D, B, x, y, eta, J, prec, verbosity):
    """
    This function computes the proximity operator using the PPXA+ algorithm
    """
    N = D.shape[1]
    x1k_old = x
    x2k_old = np.dot(D, x1k_old)
    D_transpose = D.transpose()
    A = LA.inv(np.eye(N) + np.dot(D_transpose, D))
    zk_old = np.dot(A, x1k_old + np.dot(D_transpose, x2k_old))
    teta = 1.9
    for j in range(J):
        y1k_old = proxB(B, x1k_old, x, teta)
        y2k_old = proxl2(x2k_old, y, eta)
        vk_old = np.dot(A, y1k_old + np.dot(D_transpose, y2k_old))
        x1k = x1k_old + 2 * vk_old - zk_old - y1k_old
        x2k = x2k_old + np.dot(D, 2 * vk_old - zk_old) - y2k_old
        zk = vk_old
        error = LA.norm(zk - zk_old)**2
        if error < prec:
            if verbosity == 1:
                print("PPXA stops at j = {0!s}".format(j))
            break
        x1k_old = x1k
        x2k_old = x2k
        zk_old = zk
    return zk, j


def pds(K, y, eta, nbiter):
    M, N = K.shape
    normK = norm2(K, N)
    tau = 1 / normK
    sigma = 0.9 / (tau * normK**2)
    ro = 1.0
    refspec = np.zeros((nbiter, 1))
    xk_old = np.ones((N, 1))
    uk_old = np.dot(K, xk_old)
    prec = 1e-6
    K_transpose = K.transpose()
    for i in range(nbiter):
        xxk = proxl1(xk_old - tau * np.dot(K_transpose, uk_old), tau)
        zk = uk_old + sigma * np.dot(K, 2 * xxk - xk_old)
        uuk = zk - sigma * proxl2(zk / sigma, y, eta)
        xk = xk_old + ro * (xxk - xk_old)
        uk = uk_old + ro * (uuk - uk_old)
        xk_norm = LA.norm(xk)**2
        uk_norm = LA.norm(uk)**2
        if (xk_norm == 0) or (uk_norm == 0):
            ex = sys.maxsize
            eu = sys.maxsize
        else:
            ex = LA.norm(xk - xk_old)**2 / xk_norm
            eu = LA.norm(uk - uk_old)**2 / uk_norm
        if ex < prec and eu < prec:
            break
        refspec[i] = ex
        xk_old = xk
        uk_old = uk
    return xk, refspec


def FB_PPXALpLq(K, y, p, q, metric, alpha, beta, eta, xi, nbiter, xtrue, J, verbosity):
    """
    This function defines the Trust region algorihtm based on Forward-Backward algorithm
    """

    # Initialization
    N = K.shape[1]
    xk_old = pds(K, y, xi, 10)[0]
    mysnr = [-10 * math.log10(np.sum((xk_old - xtrue)**2) / np.sum(xtrue**2))]
    fcost = [Fcost(xk_old, alpha, beta, eta, p, q)]
    gamma = 1
    prec = 1e-12
    BWhile = []
    Time = []
    # Bwhile = np.zeros((nbiter,1))
    # fcost = np.zeros((nbiter,1))
    J = J  # ppxa max iterations
    # metric 0: Lip constant, 1: FBVM without TR, 2: FBVM-TR
    L = ComputeLipschitz(alpha, beta, eta, p, q, N)

    # Algorithm
    for k in range(nbiter):
        if k % 100 == 0:
            print("it = {!s} : fcost = {!s} \n".format(k, fcost[k - 1]))
        start_time = time.time()

        if metric == 0:
            A = L * np.ones((N, 1))
            B = A / gamma
            xxk = xk_old - (1 / B) * gradlplq(xk_old, alpha, beta, eta, p, q)
            xk = proxPPXAplus(K, B, xxk, y, xi, J, prec, verbosity)[0]
            BWhile.append(0)
        elif metric == 1:
            A = condlplq(xk, alpha, beta, eta, p, q, 0)
            B = A / gamma
            xxk = xk_old - (1 / B) * gradlplq(xk_old, alpha, beta, eta, p, q)
            xk = proxPPXAplus(K, B, xxk, y, xi, J, prec, verbosity)[0]
            BWhile.append(0)
        else:
            ro = np.sum(np.abs(xk_old**q))**(1 / q)
            bwhile = 0
            while True:
                A = condlplq(xk_old, alpha, beta, eta, p, q, ro)
                B = A / gamma
                xxk = xk_old - (1 / B) * gradlplq(xk_old, alpha, beta, eta, p, q)
                xk = proxPPXAplus(K, B, xxk, y, xi, J, prec, verbosity)

                if np.sum(np.abs(xk)**q)**(1 / q) < ro:
                    ro = ro / 2
                    bwhile = bwhile + 1
                else:
                    break
            BWhile.append(bwhile)

        end_time = time.time()
        Time.append(end_time - start_time)
        mysnr.append(-10 * math.log10(np.sum((xk - xtrue)**2) / np.sum(xtrue**2)))
        fcost.append(Fcost(xk, alpha, beta, eta, p, q))
        error = LA.norm(xk - xk_old)**2 / LA.norm(xk_old)**2
        if error < prec:
            break
        xk_old = xk
    return xk_old, np.array(fcost), np.array(BWhile), np.array(Time), np.array(mysnr)
