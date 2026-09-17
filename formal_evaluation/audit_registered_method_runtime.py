"""Read only the selected method's authoritative runtime requirements."""
import json,shlex
from pathlib import Path

def audit(run,method,node,remote):
    registry=json.loads((Path(__file__).parent/'config/baseline_runtime_registry_dsw.json').read_text());spec=registry['methods'][method]
    paths=list(registry.get('shared_required_paths',[]))+list(spec.get('required_paths',[]))
    paths += [{'role':k,'path':spec[k]} for k in ['python','source_root','wilor_python'] if spec.get(k)]
    script='''import json,sys
from pathlib import Path
values=[]
for r in json.loads(sys.argv[1]):
 try:
  p=Path(r['path']);stat=p.stat();ok=('bytes' not in r or stat.st_size==r['bytes'])
  if p.is_file():
   with p.open('rb') as f:ok=ok and bool(f.read(1))
  values.append(dict(r,ok=ok))
 except OSError as e:values.append(dict(r,ok=False,error=str(e)))
print(json.dumps(dict(ok=all(r['ok'] for r in values),paths=values)))'''
    result=remote(run,node,'python3 -c '+shlex.quote(script)+' '+shlex.quote(json.dumps(paths)),timeout=120)
    if result.returncode:raise RuntimeError(result.stderr)
    return dict(json.loads(result.stdout),run_id=run['run_id'],node=node,method=method)
