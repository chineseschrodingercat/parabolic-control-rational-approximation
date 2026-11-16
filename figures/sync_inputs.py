"""Refresh figure inputs from the reproducible experiment outputs."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
for folder,name in [('robin','time_step_results.json'),('robin','prior_results.json'),('boundary_zero','results.json')]:
    dst=root/'inputs'/folder/name
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root.parent/'experiments'/folder/name,dst)
print('Updated all three figure input files.')
