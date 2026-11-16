"""Additional fixed-dictionary mesh evidence, separate from the theorem dictionary."""
from pathlib import Path
import json
import scipy.linalg as la
import run_experiments as ex

ROOT=Path(__file__).resolve().parent
rows=[]
for alpha in [1.,.001]:
    for nx in [40,160,640]:
        mo=ex.model(nx,alpha,1/320,beta=.1,xa=.41577927256,xb=.91577927256)
        rows.append(dict(alpha=alpha,beta=.1,xa=.41577927256,xb=.91577927256,nx=nx,
            F=float(la.norm(ex.factor(mo['F'],ex.MU24),2)),G=float(la.norm(ex.factor(mo['G'],ex.MU24),2)),diagnostics=mo['diagnostics']))
        print(rows[-1],flush=True)
assert max(max(r['diagnostics'].values()) for r in rows)<1e-7
(ROOT/'fixed24_grid_results.json').write_text(json.dumps(dict(passed=True,tau=1/320,mesh='uniform conforming P1',dictionary='fixed 24 poles',scope='Finite grid evidence only, not the unresolved Robin full-box theorem.',rows=rows),indent=2),encoding='utf-8')
