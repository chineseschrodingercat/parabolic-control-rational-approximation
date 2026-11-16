"""Boundary-zero family: two-precision invariant graphs and exhaustive block search."""
from __future__ import annotations

import os
for env_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[env_key] = "1"
from pathlib import Path
from fractions import Fraction as Fr
import hashlib
import json
import platform
import time
import numpy as np
import scipy
import scipy.linalg as sla
import mpmath as mp

ROOT = Path(__file__).resolve().parent
ALPHAS = ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8"]
DIGITS = [80, 120]
EPSILON = "0.01"



def dump(path, obj):
    (ROOT / path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def conv(a, b):
    out = [Fr(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


def add(a, b):
    return [(a[j] if j < len(a) else Fr(0)) + (b[j] if j < len(b) else Fr(0)) for j in range(max(len(a), len(b)))]


def exact_polynomial(alpha):
    D = [Fr(1)]
    for lam in [1, 4, 9]:
        D = conv(D, [1+Fr(lam, 100), -Fr(1)])
    q = [Fr(53,60), -Fr(184,75), Fr(257,100)]
    normq2 = sum(x*x for x in q)
    k2 = Fr(36,49)/normq2
    nbase = [Fr(1), -Fr(999,500), Fr(1)]
    weighted = [Fr(0)]
    aa = [1+Fr(x,100) for x in [1,4,9]]
    for i in range(3):
        p = [q[i]]
        for j in range(3):
            if i != j:
                p = conv(p, [aa[j], -Fr(1)])
        weighted = add(weighted, p)
    assert weighted == nbase
    # z^3 N(1/z) = k * z * reverse_2(nbase), not k * reverse_2(nbase).
    dd = [alpha*x for x in conv(D, list(reversed(D)))]
    nn = [Fr(1,10000)*k2*x for x in conv(nbase, [Fr(0)]+list(reversed(nbase)))]
    correct = add(dd, nn)
    faulty = add(dd, [Fr(1,10000)*k2*x for x in conv(nbase, list(reversed(nbase)))])
    assert len(correct) == 7 and correct == list(reversed(correct))
    assert faulty != list(reversed(faulty))
    return correct, faulty, {"D_ascending": [str(x) for x in D], "N_over_k_ascending": [str(x) for x in nbase], "k_squared": str(k2), "b_Lambda_inverse_b": str(Fr(36,49)*sum(Fr(1,x) for x in [1,4,9]))}


def mfr(x):
    return mp.mpf(x.numerator)/x.denominator


def realmat(X):
    return mp.matrix([[mp.re(X[i,j]) for j in range(X.cols)] for i in range(X.rows)])


def norm2(X):
    if not X.rows:
        return mp.mpf(0)
    gram = realmat(X.T*X)
    gram = (gram+gram.T)/2
    return mp.sqrt(max(mp.mpf(0), max(mp.eigsy(gram, eigvals_only=True))))


def relnorm(X, reference):
    return mp.norm(X)/max(mp.mpf(1), mp.norm(reference))


def power(X, n):
    assert n >= 0
    Y = mp.eye(X.rows)
    while n:
        if n & 1:
            Y = Y*X
        X = X*X
        n //= 2
    return Y


def constants(alpha, tau):
    r = 1/(2*alpha)
    chi = 1+4*r/3
    t0 = 1+r/2
    a0, b0 = chi-1, chi
    s0 = (-a0+mp.sqrt(a0*a0+mp.mpf(3)/2*b0*(a0+b0)))/b0
    sstar = min(2*r, s0)
    S = 2*(a0+b0*sstar)/(2-2*sstar+mp.mpf(4)/3*sstar*sstar)
    ctheta = 1/mp.sqrt(1+S*S)
    # -log q = atanh(4 ctheta/5); avoids cancellation q approximately 1.
    minus_log_q = mp.atanh(4*ctheta/5)
    q = mp.exp(-minus_log_q)
    m = 5*tau/(chi*(16+5*tau))
    J = max(1,int(mp.ceil(mp.log(chi/m)/mp.log(2))))
    Cstar = (1+mp.sqrt(2))*mp.sqrt(chi)*t0*t0
    ksuff = int(mp.ceil(mp.log(Cstar/mp.mpf(EPSILON))/minus_log_q))
    shifts = [m*2**j for j in range(J)]
    poles = [(1-s)/(1+s) for s in shifts]
    return dict(r=r, chi=chi, t0=t0, S=S, m=m, J=J, Cstar=Cstar,
                q=q, minus_log_q=minus_log_q, sufficient_blocks=ksuff,
                sufficient_degree=ksuff*J, shifts=shifts, poles=poles)


def block(X, shifts):
    I = mp.eye(X.rows)
    Z = (I-X)*(I+X)**-1
    B = I.copy()
    for s in shifts:
        B = (s*I-Z)*(s*I+Z)**-1*B
    return B


def scalar_block(z, poles):
    factors = [(z-mu)/(1-mu*z) for mu in poles]
    return mp.fprod(factors), sum(mp.log(abs(t)) for t in factors)


def poly_relative_residual(coeffs, z):
    cc = [mfr(x) for x in coeffs]
    return abs(mp.polyval(list(reversed(cc)),z))/sum(abs(x*z**j) for j,x in enumerate(cc))


def compute_case(astr, dps):
    mp.mp.dps = dps
    alpha, tau = mp.mpf(astr), mp.mpf(1)/100
    I = mp.eye(3)
    A = mp.diag([1+tau*x for x in [1,4,9]])
    R = A**-1
    b = mp.matrix([mp.mpf(6)/7]*3)
    qvec = mp.matrix([mp.mpf(53)/60, -mp.mpf(184)/75, mp.mpf(257)/100])
    c = qvec/mp.norm(qvec)
    Q, BB = c*c.T, b*b.T
    # [x_(k-1);p_k] -> [x_k;p_(k+1)], directly from the two KKT equations.
    H = mp.matrix(6)
    blocks = [R, tau/alpha*R*BB, tau*Q*R, A+tau*tau/alpha*Q*R*BB]
    for ii in range(2):
        for jj in range(2):
            for i in range(3):
                for j in range(3):
                    H[3*ii+i,3*jj+j] = blocks[2*ii+jj][i,j]
    vals, vecs = mp.eig(H)
    order = sorted(range(6),key=lambda j: abs(vals[j]))
    st, un = order[:3], order[3:]
    assert all(abs(vals[j]) < 1 for j in st) and all(abs(vals[j]) > 1 for j in un)
    U = mp.matrix([[vecs[i,j] for j in st] for i in range(3)])
    V = mp.matrix([[vecs[i+3,j] for j in st] for i in range(3)])
    Uu = mp.matrix([[vecs[i,j] for j in un] for i in range(3)])
    Vu = mp.matrix([[vecs[i+3,j] for j in un] for i in range(3)])
    Piraw, Sigraw = -V*U**-1, Uu*Vu**-1
    Pi = (realmat(Piraw)+realmat(Piraw).T)/2
    Sigma = (realmat(Sigraw)+realmat(Sigraw).T)/2
    F = R-tau/alpha*R*BB*Pi
    G = R*(I-tau*Q*Sigma)
    K = b.T*Pi/alpha
    T = I+Pi*Sigma
    Ffromeig = U*mp.diag([vals[j] for j in st])*U**-1
    Gfromeig = Vu*mp.diag([1/vals[j] for j in un])*Vu**-1
    correct, faulty, exact_info = exact_polynomial(Fr(astr))
    diagnostics = dict(
        primal_graph_imaginary_relative=relnorm(Piraw-realmat(Piraw),Piraw),
        dual_graph_imaginary_relative=relnorm(Sigraw-realmat(Sigraw),Sigraw),
        primal_graph_symmetry_relative=relnorm(Piraw-Piraw.T,Piraw),
        dual_graph_symmetry_relative=relnorm(Sigraw-Sigraw.T,Sigraw),
        primal_bellman_relative=relnorm(Pi-F.T*Pi*F-tau*(F.T*Q*F+alpha*K.T*K),Pi),
        dual_bellman_relative=relnorm(Sigma-G.T*Sigma*G-tau*(Sigma*Q*Sigma+G.T*BB*G/alpha),Sigma),
        paired_graph_relative=relnorm(F.T*T-T*G,T*G),
        primal_eigenstep_relative=relnorm(F-Ffromeig,F),
        dual_eigenstep_relative=relnorm(G-Gfromeig,G),
        characteristic_relative=max(poly_relative_residual(correct,z) for z in vals),
        faulty_characteristic_stable_max_relative=max(poly_relative_residual(faulty,vals[j]) for j in st),
        reciprocal_root_error=max(min(abs(vals[j]*vals[k]-1) for k in un) for j in st),
        min_Pi_eigenvalue=min(mp.eigsy(Pi,eigvals_only=True)),
        min_Sigma_eigenvalue=min(mp.eigsy(Sigma,eigvals_only=True)),
    )
    co = constants(alpha,tau)
    Bf, Bg = block(F,co["shifts"]), block(G,co["shifts"])
    scalar_results = [scalar_block(vals[j],co["poles"]) for j in st]
    log_rho = max(x[1] for x in scalar_results)
    rho = mp.exp(log_rho)
    knec = max(1,int(mp.ceil(mp.log(mp.mpf(EPSILON))/log_rho)))
    assert rho < 1 and mp.exp((knec-1)*log_rho) > mp.mpf(EPSILON)
    # No monotonicity assumption: all k<knec are excluded by the spectrum;
    # every k from knec through the first successful block is evaluated.
    Pf, Pg = power(Bf,knec), power(Bg,knec)
    all_scan = []
    k = knec
    while True:
        nf, ng = norm2(Pf), norm2(Pg)
        all_scan.append((k,nf,ng))
        if max(nf,ng) <= mp.mpf(EPSILON):
            break
        assert k < 1000000, "unexpectedly long block search"
        Pf, Pg = Pf*Bf, Pg*Bg
        k += 1
    nf_prev, ng_prev = norm2(power(Bf,k-1)), norm2(power(Bg,k-1))
    nf_next, ng_next = norm2(Pf*Bf), norm2(Pg*Bg)
    assert max(nf_prev,ng_prev) > mp.mpf(EPSILON)
    # Reconstruct the successful power directly from the independently obtained
    # invariant-graph eigenbasis, rather than reusing the block recursion.
    direct_f = U*mp.diag([x[0]**k for x in scalar_results])*U**-1
    dual_scalars = [scalar_block(1/vals[j],co["poles"])[0] for j in un]
    direct_g = Vu*mp.diag([x**k for x in dual_scalars])*Vu**-1
    diagnostics["successful_F_power_eigen_relative"] = relnorm(Pf-direct_f,Pf)
    diagnostics["successful_G_power_eigen_relative"] = relnorm(Pg-direct_g,Pg)
    zcomplex = max([vals[j] for j in st],key=lambda z: mp.im(z))
    cayley_slope = max(abs(mp.im((1-vals[j])/(1+vals[j])))/mp.re((1-vals[j])/(1+vals[j])) for j in st)
    h = mp.mpf(999)/1000
    z0 = h+mp.j*mp.sqrt(1-h*h)
    kN = mp.mpf(6)/(7*mp.norm(qvec))
    Klimit = abs(mp.fprod([A[i,i]-z0 for i in range(3)]))/(tau*abs(kN*(2*z0-2*h)))
    out = dict(alpha=alpha, digits=dps, constants=co, spectral_radius_block=rho,
               log_spectral_radius_block=log_rho, spectral_necessary_blocks=knec,
               spectral_necessary_degree=knec*co["J"], actual_blocks=k,
               actual_degree=k*co["J"], norm_F_at_actual=nf, norm_G_at_actual=ng,
               norm_F_at_previous_block=nf_prev, norm_G_at_previous_block=ng_prev,
               norm_F_at_next_block=nf_next, norm_G_at_next_block=ng_next,
               scan_evaluated_blocks=len(all_scan),
               scanned_block_norms=[dict(block=kk,F=ff,G=gg) for kk,ff,gg in all_scan],
               minimum_failed_norm_in_scan=min([max(a,b) for _,a,b in all_scan[:-1]],default=None),
               one_block_norm_F=norm2(Bf), one_block_norm_G=norm2(Bg),
               complex_stable_root=zcomplex, cayley_spectral_slope=cayley_slope,
               slope_times_sqrt_alpha=cayley_slope*mp.sqrt(alpha),
               boundary_constant_K=Klimit, asymptotic_slope_constant=mp.sqrt(1-h*h)/Klimit,
               stable_roots=[vals[j] for j in st], unstable_roots=[vals[j] for j in un],
               matrices=dict(F=F,G=G,Pi=Pi,Sigma=Sigma,B_F=Bf,B_G=Bg),
               diagnostics=diagnostics, exact_polynomial=exact_info,
               characteristic_ascending_exact=[str(x) for x in correct])
    return out


def jsonify(x, digits=55):
    if isinstance(x,dict):
        return {k:jsonify(v,digits) for k,v in x.items()}
    if isinstance(x,(list,tuple)):
        return [jsonify(v,digits) for v in x]
    if isinstance(x,mp.matrix):
        return [[jsonify(x[i,j],digits) for j in range(x.cols)] for i in range(x.rows)]
    if isinstance(x,mp.mpc):
        return {"real":mp.nstr(mp.re(x),digits),"imag":mp.nstr(mp.im(x),digits)}
    if isinstance(x,mp.mpf):
        return mp.nstr(x,digits)
    return x


def scipy_case(alpha):
    tau = .01
    R = np.diag(1/(1+tau*np.array([1.,4.,9.])))
    b = np.full((3,1),6/7)
    c = np.array([[53/60],[-184/75],[257/100]])
    c /= np.linalg.norm(c)
    Q = c@c.T
    def primal(inp, obs, ctrl_weight):
        D = tau*R@inp
        Qd = R@obs@R
        S = R@obs@D
        Rc = ctrl_weight*np.eye(1)+D.T@obs@D
        P = sla.solve_discrete_are(R,D,Qd,Rc,s=S)
        K = sla.solve(Rc+D.T@P@D,D.T@P@R+S.T,assume_a="pos")
        return R-D@K, tau*P
    F, Pi = primal(b,Q,alpha)
    G, Sigma = primal(c,b@b.T/alpha,1.)
    return dict(F=F,G=G,Pi=Pi,Sigma=Sigma)


def precision_compare(low,high):
    mp.mp.dps = 120
    out = {}
    for name in ["F","G","Pi","Sigma","B_F","B_G"]:
        a,b = low["matrices"][name],high["matrices"][name]
        out[name+"_relative_80_120"] = relnorm(a-b,b)
    for name in ["spectral_radius_block","norm_F_at_actual","norm_G_at_actual","norm_F_at_previous_block","norm_G_at_previous_block"]:
        out[name+"_absolute_80_120"] = abs(low[name]-high[name])
    names = ["actual_degree","spectral_necessary_degree","actual_blocks","spectral_necessary_blocks"]
    out["integer_degrees_identical"] = all(low[key]==high[key] for key in names) and low["constants"]["sufficient_degree"]==high["constants"]["sufficient_degree"]
    assert out["integer_degrees_identical"]
    return out

PARAMETERS = {'lambda': '1', 'gamma': '1', 'eta': '1', 'tau': '1/100', 'epsilon': '1/100', 'alpha_equals_alpha_floor': True, 'Lambda_diagonal': [1, 4, 9], 'b_each': '6/7', 'q': ['53/60', '-184/75', '257/100'], 'c': 'q / Euclidean_norm(q)', 'Q': 'c c^T'}
DEGREE_CONVENTIONS = {'spectral': 'J ceil(log(epsilon)/max_eigenvalue log|Psi_J(z)|); necessary for this constructed dictionary only', 'actual': 'minimum kJ over integer k>=1 for which max(norm_2(B_F^k),norm_2(B_G^k)) <= epsilon', 'sufficient': 'J ceil(log(Cstar/epsilon)/atanh(4/(5 sqrt(1+S^2))))', 'search': 'All k<k_spec excluded by eigenvalues; exhaustive k_spec,...,k_actual in arbitrary precision. Nonmonotonicity is allowed.'}

def main():
    start = time.perf_counter()
    computations = {}
    for dps in DIGITS:
        computations[dps] = []
        for astr in ALPHAS:
            row = compute_case(astr,dps)
            computations[dps].append(row)
            print(f"digits={dps} alpha={astr} J={row['constants']['J']} spectral={row['spectral_necessary_degree']} actual={row['actual_degree']} sufficient={row['constants']['sufficient_degree']}",flush=True)
    comparisons, scipy_checks = [], []
    rows = computations[120]
    for low,high in zip(computations[80],rows):
        comparisons.append({"alpha":high["alpha"],**precision_compare(low,high)})
        sc = scipy_case(float(high["alpha"]))
        er = {"alpha":float(high["alpha"])}
        for key in ["F","G","Pi","Sigma"]:
            exact = np.array(high["matrices"][key].tolist(),dtype=float)
            er[key+"_relative_double_120"] = float(np.linalg.norm(sc[key]-exact)/max(1.,np.linalg.norm(exact)))
        scipy_checks.append(er)
    fitted = {}
    for label,include in [("all_1e-2_through_1e-8",rows),("tail_1e-4_through_1e-8",rows[2:])]:
        xx = np.log([1/float(r["alpha"]) for r in include])
        fitted[label] = {"convention":"OLS log(d) = intercept + exponent log(1/alpha); complete-block degrees; equally weighted declared alpha points", "alpha_values":[str(r["alpha"]) for r in include]}
        for key in ["actual_degree","spectral_necessary_degree"]:
            slope,intercept = np.polyfit(xx,np.log([r[key] for r in include]),1)
            fitted[label][key] = {"exponent":float(slope),"intercept":float(intercept)}
        slope,intercept=np.polyfit(xx,np.log([r["constants"]["sufficient_degree"] for r in include]),1)
        fitted[label]["sufficient_degree"]={"exponent":float(slope),"intercept":float(intercept)}
    result = {
        "scope": "Finite-dimensional boundary-zero family, Euclidean operator norms.",
        "parameters": PARAMETERS,
        "degree_conventions": DEGREE_CONVENTIONS,
        "rows": rows, "precision_comparisons": comparisons,
        "scipy_independent_primal_and_dual_checks": scipy_checks,
        "descriptive_log_log_fits": fitted,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "mpmath": mp.__version__},
        "elapsed_seconds": time.perf_counter()-start
    }
    dump("results.json", jsonify(result))
    residual_names = [x for x in rows[0]["diagnostics"] if x not in ["faulty_characteristic_stable_max_relative","min_Pi_eigenvalue","min_Sigma_eigenvalue"]]
    maxres=max(r["diagnostics"][key] for r in rows for key in residual_names)
    maxprec=max(v for r in comparisons for key,v in r.items() if key.endswith("80_120"))
    maxscmoderate=max(v for r in scipy_checks if r["alpha"]>=1e-4 for key,v in r.items() if key!="alpha")
    evidence={"passed":False,"checks":{"all_integer_degrees_match_80_120":all(x["integer_degrees_identical"] for x in comparisons),"120_digit_residuals_below_1e-95":maxres<mp.mpf("1e-95"),"80_120_comparison_below_1e-55":maxprec<mp.mpf("1e-55"),"independent_scipy_moderate_alpha_below_1e-8":maxscmoderate<1e-8,"positive_value_operators":all(r["diagnostics"]["min_Pi_eigenvalue"]>0 and r["diagnostics"]["min_Sigma_eigenvalue"]>0 for r in rows),"first_crossing_failed_predecessor":all(max(r["norm_F_at_previous_block"],r["norm_G_at_previous_block"])>mp.mpf(EPSILON) for r in rows),"first_crossing_successful":all(max(r["norm_F_at_actual"],r["norm_G_at_actual"])<=mp.mpf(EPSILON) for r in rows),"block_search_exhaustive_after_spectral_exclusion":all(r["scan_evaluated_blocks"]==r["actual_blocks"]-r["spectral_necessary_blocks"]+1 for r in rows),"exact_reciprocal_polynomial":True},"maximum_120_digit_residual":maxres,"maximum_80_120_difference":maxprec,"maximum_moderate_scipy_difference":maxscmoderate,"scope":"Executed numerical diagnostics, not rigorous interval enclosures. Two precisions, independently formulated SciPy primal/dual Riccati equations, exact reciprocal coefficients, invariant eigenbasis powers, and exhaustive complete-block crossing."}
    evidence["passed"]=all(evidence["checks"].values())
    dump("check_evidence.json",jsonify(evidence))
    assert evidence["passed"],jsonify(evidence)
    print("ALL_CHECKS_PASSED", flush=True)

if __name__ == "__main__":
    main()
