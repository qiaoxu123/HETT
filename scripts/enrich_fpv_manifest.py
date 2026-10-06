#!/usr/bin/env python3
"""Add explicit camera calibration and relative-yaw metadata to render records."""
import argparse, json, math
from pathlib import Path

def cam(pose, pitch_deg, width, height):
    f=(width/2)/math.tan(math.radians(90)/2)
    yaw=float(pose[3]);dx=.5*math.cos(yaw);dy=.5*math.sin(yaw)
    return {"intrinsics":{"width":width,"height":height,"horizontal_fov_deg":90.0,"fx_px":f,"fy_px":f,"cx_px":width/2,"cy_px":height/2},
            "extrinsics":{"camera_name":"front_0","vehicle_to_camera_xyz_ned_m":[.5,0.0,0.0],"camera_world_xy_enu":[float(pose[0]+dx),float(pose[1]+dy)],"camera_pitch_deg":float(pitch_deg),"camera_yaw_rad":yaw,"transform":"CityNav ENU (x,y,z,yaw) -> AirSim NED (x,-y,-z,pitch,roll,-yaw)"}}

def enrich(obs,pitch):
    pose=obs.get('pose')
    if not pose:return
    image=Path(obs['image']); width=height=224
    try:
        import cv2
        frame=cv2.imread(str(image))
        if frame is not None:height,width=frame.shape[:2]
    except Exception:pass
    obs.update(cam(pose,pitch,width,height))
    for c in obs.get('candidates',[]):
        xy=c.get('world_xy')
        if xy:
            c['relative_yaw_rad']=float((math.atan2(float(xy[1])-float(pose[1]),float(xy[0])-float(pose[0]))-float(pose[3])+math.pi)%(2*math.pi)-math.pi)
        c.setdefault('occlusion_ratio',None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();path=a.dataset/'manifest.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
    for row in rows:
        for name,obs in row.get('view_observations',{}).items():enrich(obs,0 if name=='fpv' else -30 if name=='oblique30' else -45)
        for obs in row.get('history_fpv',[]):enrich(obs,0)
    with path.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'rows':len(rows),'enriched_camera_records':sum(len(r.get('view_observations',{}))+len(r.get('history_fpv',[])) for r in rows),'manifest':str(path)},indent=2))
if __name__=='__main__':main()
