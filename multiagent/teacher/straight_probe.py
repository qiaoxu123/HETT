"""Training-only oracle teacher geometry and deterministic screening subset."""
import math
import random
from collections import defaultdict
from multiagent.space import Pose4D

def straight_teacher(start, goal, stride=5.):
    if stride<=0: raise ValueError('stride must be positive')
    dx,dy=goal.x-start.x,goal.y-start.y
    distance=math.hypot(dx,dy)
    if distance<1e-8: return [start]
    heading=math.atan2(dy,dx)
    path=[start]
    for step in range(1,math.ceil(distance/stride)+1):
        fraction=min(step*stride/distance,1.)
        path.append(Pose4D(start.x+fraction*dx,start.y+fraction*dy,start.z,heading))
    return path

def balanced_subset(trajectories, count, seed):
    if not count: return trajectories
    groups=defaultdict(list)
    for t in trajectories: groups[t.map_name].append(t)
    rng=random.Random(seed)
    for name in sorted(groups): rng.shuffle(groups[name])
    selected=[]
    while len(selected)<count:
        added=False
        for name in sorted(groups):
            if groups[name] and len(selected)<count:
                selected.append(groups[name].pop());added=True
        if not added: break
    return selected
