"""Read-only audit of 300-frame coverage for the frozen 2D candidates."""
import argparse, json, re
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    out=[]
    for r in rows:
        c=int(re.search(r'\d+',str(r['center_frame_id'])).group()); lo,hi=r['start_frame'],r['end_frame']
        # This worker only records auditable source metadata. It never guesses missing frames.
        out.append({**r,'required_frame_ids':list(range(lo,hi+1)),'coverage_status':'unresolved','missing_frame_ids':list(range(lo,hi+1)),'audit_reason':'source indices must be supplied on DSW'})
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text('\n'.join(json.dumps(x) for x in out)+'\n')
    print(json.dumps({'status':'coverage_audit_complete','candidates':len(out),'renderable':0,'unresolved':len(out)}))
if __name__=='__main__': main()
