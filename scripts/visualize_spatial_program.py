#!/usr/bin/env python3
"""Render auditable top-down records from frozen inference traces."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('--debug-records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);records=json.loads(a.debug_records.read_text())
 manifest=[]
 for row in records:
  fig,ax=plt.subplots(figsize=(7,7));c=np.asarray([x['xy'] for x in row['candidates']]);b=np.asarray([x['b0'] for x in row['candidates']]);ax.scatter(c[:,0],c[:,1],s=80+500*b,c=np.arange(len(c)),cmap='viridis',label='B0 Top-K')
  for i,p in enumerate(c):ax.text(p[0],p[1],str(i+1),fontsize=7)
  for anchor in row['anchors']:
   contour=np.asarray(anchor['contour']);ax.plot(*np.vstack([contour,contour[:1]]).T,color='tab:red',lw=2);ax.scatter(*anchor['centroid'],marker='*',s=160,color='tab:red')
  start=np.asarray(row['start_pose'][:2]);axis=np.asarray(row['trace'].get('axis',{}).get('vector',[1,0]));ax.arrow(*start,*(axis*30),width=.8,color='tab:orange');ax.scatter(*start,marker='^',s=100,color='black')
  ax.set_aspect('equal');ax.set_title(row['instruction']+'\n'+row['failure'],fontsize=9);ax.legend(fontsize=7);fig.tight_layout();name=f"{row['split']}_{row['episode_index']:05d}.png";fig.savefig(a.output/name,dpi=110);plt.close(fig);manifest.append({'file':name,'split':row['split'],'episode_index':row['episode_index'],'failure':row['failure']})
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'images':len(manifest),'splits':{s:sum(x['split']==s for x in manifest) for s in ('val_seen','val_unseen')}},indent=2))
if __name__=='__main__':main()
