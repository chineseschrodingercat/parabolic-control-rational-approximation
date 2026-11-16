"""Recheck the exact scalar subrange certificate using only this folder."""
from fractions import Fraction as F
from pathlib import Path
import json
import research

root=Path(__file__).resolve().parent
result=research.certificate('one_percent_full_beta_subclass',F(53,100),F(1,10),F(53,20),F(407,250),F(303,1000),F(1,100))
expected=json.loads((root/'verified_certificate.json').read_text())['certificate']
assert all(result[k]==v for k,v in expected.items() if k!='seconds')
print(json.dumps({'passed':True,'scalar_upper':result['scalar_upper_rational'],'both_upper':result['both_mass_operator_upper_rational'],'cells':result['circle_cells']+result['parabola_cells']},indent=2))
