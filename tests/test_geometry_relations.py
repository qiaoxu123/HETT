from multiagent.geometry_reasoning.relations import directional,near,sequence,between
def test_directions():
 h=(1,0);a=(0,0);assert directional((0,-10),a,h,'right_of')>directional((0,10),a,h,'right_of');assert sequence((10,0),a,h,'past')>sequence((-10,0),a,h,'past')
def test_near_between():assert near((1,0),(0,0))>near((50,0),(0,0));assert between((5,1),(0,0),(10,0))>between((20,10),(0,0),(10,0))
