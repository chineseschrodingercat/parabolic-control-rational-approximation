"""Exact rational counterexample for the abstract form class, NOT the Robin class.

All assertion arithmetic uses fractions.Fraction.  Decimal displays are outward
rounded rational intervals. No Riccati solver is involved in this certificate.
"""
from fractions import Fraction as Q
from pathlib import Path
import json, hashlib, sys
sys.set_int_max_str_digits(1000000)
ROOT=Path(__file__).resolve().parent
def mat(x): return [[Q(v) for v in row] for row in x]
I=mat([[1,0],[0,1]])
Z=mat([[0,0],[0,0]])
def add(A,B): return [[a+b for a,b in zip(ar,br)] for ar,br in zip(A,B)]
def neg(A): return [[-v for v in r] for r in A]
def sub(A,B): return add(A,neg(B))
def mul(A,B): return [[sum((A[i][k]*B[k][j] for k in range(len(B))),Q(0)) for j in range(len(B[0]))] for i in range(len(A))]
def scale(c,A): return [[c*v for v in r] for r in A]
def tr(A): return [list(r) for r in zip(*A)]
def det(A): return A[0][0]*A[1][1]-A[0][1]*A[1][0]
def inv(A):
    d=det(A); assert d!=0
    return scale(1/d,[[A[1][1],-A[0][1]],[-A[1][0],A[0][0]]])
def pd(A): return A==tr(A) and A[0][0]>0 and det(A)>0
def encode(A): return [[str(v) for v in r] for r in A]
def interval(x,n=12):
    k=x.numerator*10**n//x.denominator
    return [str(Q(k,10**n)),str(Q(k+1,10**n))]
def solve(A,b):
    A=[list(r)+[v] for r,v in zip(A,b)]
    n=len(A)
    for j in range(n):
        p=next(i for i in range(j,n) if A[i][j])
        A[j],A[p]=A[p],A[j]
        q=A[j][j];A[j]=[v/q for v in A[j]]
        for i in range(n):
            if i!=j:
                q=A[i][j];A[i]=[a-q*b for a,b in zip(A[i],A[j])]
    return [r[-1] for r in A]
tau=Q(1,320);alpha=Q(1,1000);eta=Q(10,11)
Lambda=mat([['53/20',0],[0,'637/100']])
b=mat([['-441/500'],['989/500']])
Pi=add(scale(Q(999,1000),mat([['18593/500000','2437/100000'],['2437/100000','1199/62500']])),scale(Q(1,100000),I))
d=alpha-tau*mul(mul(tr(b),Pi),b)[0][0]
w=scale(1/d,mul(mul(add(I,scale(tau,Lambda)),Pi),b))
Qcost=add(add(mul(Lambda,Pi),mul(Pi,Lambda)),add(scale(tau,mul(mul(Lambda,Pi),Lambda)),scale(d,mul(w,tr(w)))))
F=inv(add(I,scale(tau,add(Lambda,mul(b,tr(w))))))
K=mul(tr(w),F)
R=inv(add(I,scale(tau,Lambda)));D=scale(tau,mul(R,b));S=add(Pi,scale(tau,Qcost))
den=tau*alpha+mul(mul(tr(D),S),D)[0][0]
K_bellman=scale(1/den,mul(mul(tr(D),S),R))
bellman=sub(mul(mul(tr(R),S),R),scale(1/den,mul(mul(mul(mul(tr(R),S),D),tr(D)),mul(S,R))))
stage=scale(tau,add(mul(mul(tr(F),Qcost),F),scale(alpha,mul(tr(K),K))))
checks={
 'positive_value':pd(Pi),'positive_input_denominator':d>0,
 'lambda_lower_bound':Lambda[0][0]>=Q(53,20) and Lambda[1][1]>=Q(53,20),
 'boundary_form_bound':pd(sub(scale(eta,Lambda),mul(b,tr(b)))),
 'positive_observation':pd(Qcost),'observation_below_identity':pd(sub(I,Qcost)),
 'feedback_identity':K==K_bellman,'closed_loop_identity':F==sub(R,mul(D,K)),
 'bellman_identity':Pi==bellman,
 'strict_stability_lyapunov_identity':sub(Pi,mul(mul(tr(F),Pi),F))==stage and pd(stage)
}
poles=json.loads((ROOT/'scalar_certificate.json').read_text(encoding='utf8'))['cases'][0]['poles_exact']
Y=I
for ps in poles:
    mu=Q(ps);Y=mul(mul(inv(sub(I,scale(mu,F))),sub(F,scale(mu,I))),Y)
first_column_sq=Y[0][0]**2+Y[1][0]**2
checks['factor_norm_strictly_above_one_quarter']=first_column_sq>Q(1,16)
# Exact symmetric Stein solve, H-F^T H F=tau K^T K.
bases=[mat([[1,0],[0,0]]),mat([[0,1],[1,0]]),mat([[0,0],[0,1]])]
positions=[(0,0),(0,1),(1,1)]
cols=[sub(E,mul(mul(tr(F),E),F)) for E in bases]
rhs=scale(tau,mul(tr(K),K))
h=solve([[C[i][j] for C in cols] for i,j in positions],[rhs[i][j] for i,j in positions])
H=mat([[h[0],h[1]],[h[1],h[2]]])
tail=mul(mul(tr(Y),H),Y)
relative_sq=(tail[0][0]+tail[1][1])/(H[0][0]+H[1][1])
checks['exact_response_stein_identity']=sub(H,mul(mul(tr(F),H),F))==rhs and pd(H)
checks['initial_response_relative_H2_error_above_ten_percent']=relative_sq>Q(1,100)
result={
 'passed':all(checks.values()),'scope':'Abstract self-adjoint form class only; not an actual Robin PDE or P1 counterexample.',
 'arithmetic':'Exact rational arithmetic for every assertion; outward rational display intervals.',
 'tau':str(tau),'alpha':str(alpha),'eta':str(eta),'lambda_lower':'53/20','gamma':'1',
 'Lambda':encode(Lambda),'b':encode(b),'Pi':encode(Pi),'d':str(d),'w':encode(w),'Q':encode(Qcost),'F':encode(F),'K':encode(K),'H_initial':encode(H),
 'fixed_poles_exact':poles,
 'factor_first_column_squared_interval':interval(first_column_sq),
 'relative_initial_response_H2_error_squared_interval':interval(relative_sq),
 'factor_matrix_outward_intervals':[[interval(v) for v in row] for row in Y],
 'exact_factor_sha256':hashlib.sha256(json.dumps(encode(Y),separators=(',',':')).encode()).hexdigest(),
 'checks':checks,
 'excluded_from_robin_by':'lambda_2=6.37 < (3*pi/2)^2; every conforming Robin subspace obeys the latter minmax lower bound.',
 'not_disproved':['Universal adaptive-pole approximation rate','Fixed 24 poles on the stated Robin parameter and mesh class']
}
(ROOT/'certificate.json').write_text(json.dumps(result,indent=2),encoding='utf8')
assert result['passed'],checks
print(json.dumps({k:result[k] for k in ['passed','factor_first_column_squared_interval','relative_initial_response_H2_error_squared_interval','exact_factor_sha256','checks']},indent=2))
