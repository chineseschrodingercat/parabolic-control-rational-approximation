"""Robin heat-control experiments. Run from any directory; outputs are beside this script."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
from fractions import Fraction
import json,time,csv,hashlib,platform
import numpy as np
import scipy
import scipy.linalg as la
from scipy.signal import lfilter
from scipy.stats import qmc
from robin_solver import RobinConfig,assemble_operators

ROOT=Path(__file__).resolve().parent
LAMBDA=np.pi**2/4
MU24=np.array([float(Fraction(s)) for s in json.loads((ROOT/'fixed24_scalar_certificate.json').read_text())['cases'][0]['poles_exact']])

def dump(name,obj):
    (ROOT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')

def constants(alpha_min,tau,improved=True):
    r=1/(2*alpha_min*LAMBDA);chi=1+4*r/3;t0=1+r/2
    a=chi-1;b=chi
    crit=(-a+np.sqrt(a*a+1.5*b*(a+b)))/b
    sarg=min(2*r,crit)
    slope=2*(a+b*sarg)/(2-2*sarg+4*sarg*sarg/3) if improved else 4*r+1.6
    ctheta=1/np.sqrt(1+slope*slope)
    q=np.sqrt((5-4*ctheta)/(5+4*ctheta))
    m=5*tau*LAMBDA/(chi*(16+5*tau*LAMBDA))
    J=max(1,int(np.ceil(np.log2(chi/m))))
    poles=(1-m*2.**np.arange(J))/(1+m*2.**np.arange(J))
    Astar=1+(2*np.log(chi)+np.log(21/5))/np.log(2)
    C0=1+np.sqrt(2);preF=C0*np.sqrt(chi);preBoth=preF*t0*t0
    C=preBoth/q;omega=-np.log(q)/Astar
    ell=1+max(0,np.log(1/(LAMBDA*tau)))
    return dict(r=r,chi=chi,t0=t0,slope=slope,q=q,m=m,J=J,poles=poles.tolist(),Astar=Astar,omega=omega,C=C,preF=preF,preBoth=preBoth,ell=ell)

def model(nx,alpha,tau,beta=1.,xa=.25,xb=.75,dual=True):
    cfg=RobinConfig(nx=nx,Nt=320,T=tau*320,alpha=alpha,beta=beta,xa=xa,xb=xb)
    ops=assemble_operators(cfg)
    M=ops.M.toarray();stiff=ops.K_beta.toarray();obs=ops.M_obs.toarray()
    lam,U=la.eigh(stiff,M,check_finite=False)
    b=U.T@ops.e0;Q=U.T@obs@U;Q=(Q+Q.T)/2
    R=np.diag(1/(1+tau*lam));D=tau*(R@b[:,None])
    Qd=R@Q@R;S=R@Q@D;Rc=alpha*np.eye(1)+D.T@Q@D
    P=la.solve_discrete_are(R,D,Qd,Rc,s=S)
    K=la.solve(Rc+D.T@P@D,D.T@P@R+S.T,assume_a='pos')
    F=R-D@K;Pi=tau*P
    pres=la.norm(Pi-F.T@Pi@F-tau*(F.T@Q@F+alpha*K.T@K),'fro')/max(1,la.norm(Pi,'fro'))
    kerr=la.norm(K-b[None,:]@Pi/alpha)/max(1,la.norm(K))
    ans={'F':F,'K':K,'Pi':Pi,'lam':lam,'b':b,'Q':Q,'R':R,'diagnostics':{'primal_bellman_relative':float(pres),'feedback_relative':float(kerr)}}
    if dual:
        H=Pi+tau*Q;h=tau*alpha+(D.T@H@D)[0,0]
        W=la.solve_discrete_lyapunov(F,D@D.T/h,method='bilinear');W=(W+W.T)/2
        Sdual=la.solve((np.eye(nx)-Pi@W).T,W.T).T;Sdual=(Sdual+Sdual.T)/2
        G=R@(np.eye(nx)-tau*Q@Sdual);T=np.eye(nx)+Pi@Sdual
        gres=la.norm(Sdual-G.T@Sdual@G-tau*(Sdual@Q@Sdual+G.T@np.outer(b,b)@G/alpha),'fro')/max(1,la.norm(Sdual,'fro'))
        graph=la.norm(F.T@T-T@G,'fro')/max(1,la.norm(T@G,'fro'))
        ans.update(G=G,Sigma=Sdual)
        ans['diagnostics'].update(dual_bellman_relative=float(gres),paired_graph_relative=float(graph))
    return ans

def factor(A,poles):
    I=np.eye(len(A));Y=I.copy()
    for p in poles:Y=la.solve(I-p*A,(A-p*I)@Y,check_finite=False)
    return Y

def curve(A,poles,maxdegree):
    I=np.eye(len(A));Y=I.copy();out=[1.]
    for k in range(maxdegree):
        p=poles[k%len(poles)]
        Y=la.solve(I-p*A,(A-p*I)@Y,check_finite=False)
        out.append(float(la.norm(Y,2)))
    return out

def block_degree(block,threshold,maxblocks=100):
    P=np.eye(len(block))
    for k in range(1,maxblocks+1):
        P=P@block
        if la.norm(P,2)<=threshold:return k
    return None

def required(c,threshold,both=True):
    pre=c['preBoth' if both else 'preF']
    return c['J']*max(0,int(np.ceil(np.log(pre/threshold)/(-np.log(c['q'])))))

def basis(n):
    x=np.stack([np.sqrt(1-p*p)*p**np.arange(n) for p in MU24],axis=1)
    for j in range(len(MU24)):
        for i in range(j):x[:,j]=lfilter([-MU24[i],1],[1,-MU24[i]],x[:,j])
    return x

def prior_experiment():
    tau=1/320;amin=.53;Astar=1/(2*np.sqrt(amin))
    scalar_delta=float(Fraction(678617503,250000000000))
    zeta=(1+np.sqrt(2))*scalar_delta*Astar
    m=float(Fraction(56951,33125));rho=1/(1+tau*m)
    windows=[320,480,640,800,1000];Phi=basis(max(windows))
    bg=[]
    for n in windows:
        c=float(la.svdvals(Phi[:n])[-1])
        bg.append(dict(n=n,c=c,ideal_rational=zeta/c/Astar,ideal_prefix=rho**n))
    params=[]
    for alpha in [.53,1.]:
        for beta in [.1,10.]:
            for width in [.1,.5]:
                for x in [0.,1-width]:params.append((alpha,beta,x,x+width,'corner'))
    for a,b,w,x in qmc.Sobol(4,scramble=True,seed=20261115).random_base2(5):
        alpha=.53+.47*a;beta=.1*100**b;width=.1+.4*w;xa=(1-width)*x
        params.append((alpha,beta,xa,xa+width,'sobol'))
    rows=[];diagnostics=[]
    for idx,(alpha,beta,xa,xb,kind) in enumerate(params):
        mo=model(40,alpha,tau,beta,xa,xb,dual=False);F=mo['F'];K=mo['K'];C=-np.sqrt(tau)*K
        H=la.solve_discrete_lyapunov(F.T,C.T@C,method='bilinear');H=(H+H.T)/2
        Y=np.eye(len(F));Gamma=[]
        for p in MU24:
            Z=la.solve(np.eye(len(F))-p*F,Y,check_finite=False)
            Gamma.append(np.sqrt(1-p*p)*(C@Z).reshape(-1));Y=(F-p*np.eye(len(F)))@Z
        Gamma=np.asarray(Gamma);tail=float(np.trace(Y.T@H@Y));energy=float(np.trace(H))
        exact_head=np.empty((max(windows),len(F)));v=C.copy()
        for k in range(len(exact_head)):exact_head[k]=v;v=v@F
        pyerr=abs(energy-np.sum(Gamma**2)-tail)/max(1,energy)
        rng=np.random.default_rng(70000+idx);noise=rng.normal(size=exact_head.shape);noise/=la.norm(noise,'fro')
        diag={**mo['diagnostics'],'pythagoras_relative':float(pyerr),'actual_projection_tail':float(np.sqrt(max(0,tail))),'class_rational_budget':zeta}
        diagnostics.append(diag)
        for level in [0.,1e-4,1e-3]:
            delta=level*Astar;noisy=exact_head+delta*noise
            for drow in bg:
                n=drow['n'];c=drow['c'];X=Phi[:n];y=noisy[:n]
                fitted=la.lstsq(X,y,lapack_driver='gelsd')[0]
                residual=float(la.norm(X@fitted-y,'fro'));r=residual+delta
                bound=np.sqrt(r*r+2*np.sqrt(1-c*c)*r*zeta+zeta*zeta)/c
                actual=np.sqrt(max(0,float(la.norm(fitted-Gamma,'fro')**2+tail)))
                Fn=np.linalg.matrix_power(F,n);prefix_tail=float(np.trace(Fn.T@H@Fn))
                prefix_actual=np.sqrt(max(0,la.norm(delta*noise[:n],'fro')**2+prefix_tail))
                prefix_bound=np.sqrt(delta*delta+(Astar*rho**n)**2)
                rows.append(dict(case=idx,kind=kind,alpha=alpha,beta=beta,xa=xa,xb=xb,n=n,noise_fraction=level,c=c,residual_over_amplitude=residual/Astar,rational_bound=bound/Astar,rational_error=actual/Astar,prefix_bound=prefix_bound/Astar,prefix_error=prefix_actual/Astar,direct_projection_error=np.sqrt(max(0,tail))/Astar,amplitude=Astar))
        if idx%12==0:print('prior case',idx,flush=True)
    checks={'rational_coverage':all(x['rational_error']<=x['rational_bound']+1e-10 for x in rows),'prefix_coverage':all(x['prefix_error']<=x['prefix_bound']+1e-10 for x in rows),'projection_budget_valid_on_instances':all(x['actual_projection_tail']<=zeta+1e-10 for x in diagnostics),'pythagoras_max':max(x['pythagoras_relative'] for x in diagnostics)}
    out=dict(description='48 independently evaluated PDE cases, 16 corners plus 32 Sobol; nx40; deterministic bounded noise; class amplitude uses unknown alpha lower bound .53',windows=bg,params=params,rows=rows,checks=checks,diagnostics=diagnostics,prior=dict(alpha_lower=amin,Astar=Astar,delta_scalar=scalar_delta,zeta=zeta,m=m,rho=rho))
    dump('prior_results.json',out);return checks

def main():
    start=time.perf_counter();rows=[];all_diag=[]
    for den in [20,80,320,1280,5120]:
        tau=1/den;mo=model(160,1,tau);c=constants(1,tau);old=constants(1,tau,False)
        maxd=6*c['J'];f=curve(mo['F'],c['poles'],maxd);g=curve(mo['G'],c['poles'],maxd)
        Bf=factor(mo['F'],c['poles']);Bg=factor(mo['G'],c['poles'])
        deg=[]
        for eps in [.01,.0001]:
            kf=block_degree(Bf,eps);kg=block_degree(Bg,eps)
            deg.append(dict(epsilon=eps,observed_F=kf*c['J'],observed_both=max(kf,kg)*c['J'],proved_F=required(c,eps,False),proved_both=required(c,eps),old_proved_F=required(old,eps,False),old_proved_both=required(old,eps),necessary_degree_real=2/np.pi**2*np.log(1/eps)*np.log(1+2/(LAMBDA*tau))))
        rows.append(dict(denominator=den,tau=tau,constants=c,old_constants=old,degrees=deg,F=f,G=g,diagnostics=mo['diagnostics']))
        all_diag.append(mo['diagnostics']);print('time step',den,'J',c['J'],'degrees',deg,flush=True)
    dump('time_step_results.json',dict(config=dict(nx=160,beta=1,xa=.25,xb=.75,alpha=1,lambda_class=LAMBDA,gamma=1,eta=1),rows=rows))
    grid=[]
    for alpha in [1.,.01]:
        c=constants(alpha,1/320)
        for nx in [40,160,640]:
            mo=model(nx,alpha,1/320);f=float(la.norm(factor(mo['F'],c['poles']),2));g=float(la.norm(factor(mo['G'],c['poles']),2))
            grid.append(dict(alpha=alpha,nx=nx,J=c['J'],F=f,G=g,diagnostics=mo['diagnostics']));all_diag.append(mo['diagnostics']);print('grid',nx,alpha,f,g,flush=True)
    dump('grid_results.json',dict(config=dict(tau=1/320,beta=1,xa=.25,xb=.75),rows=grid))
    alphas=[]
    for alpha in [100.,10.,1.,.5,.1,.01,.001]:
        c=constants(alpha,1/320);old=constants(alpha,1/320,False);mo=model(160,alpha,1/320)
        bf=factor(mo['F'],c['poles']);bg=factor(mo['G'],c['poles'])
        alphas.append(dict(alpha_lower=alpha,improved=c,original=old,observed_F=c['J']*block_degree(bf,.01),observed_both=c['J']*max(block_degree(bf,.01),block_degree(bg,.01)),proved_F=required(c,.01,False),proved_both=required(c,.01),old_proved_F=required(old,.01,False),old_proved_both=required(old,.01),simplified_F_degree=int(np.ceil(c['ell']*np.log(c['preF']/c['q']/.01)/c['omega'])),diagnostics=mo['diagnostics']))
        all_diag.append(mo['diagnostics']);print('alpha',alpha,'J',c['J'],flush=True)
    dump('alpha_results.json',dict(config=dict(nx=160,tau=1/320,beta=1,xa=.25,xb=.75),rows=alphas))
    pc=prior_experiment()
    diagmax={k:max(d.get(k,0.) for d in all_diag) for k in all_diag[0]}
    checks={'matrix_identity_residuals':max(diagmax.values())<1e-7,'prior_rational_coverage':pc['rational_coverage'],'prior_prefix_coverage':pc['prefix_coverage'],'prior_projection_bound':pc['projection_budget_valid_on_instances'],'prior_pythagoras':pc['pythagoras_max']<1e-9}
    dump('execution_checks.json',dict(passed=all(checks.values()),checks=checks,max_matrix_residuals=diagmax,prior=pc,seconds=time.perf_counter()-start,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),scientific_scope='Finite numerical experiments, not global certificates. Analytical bounds use proved class constants.'))
    assert all(checks.values()),checks
    print('ALL_CHECKS_PASSED',flush=True)

if __name__=='__main__':main()
