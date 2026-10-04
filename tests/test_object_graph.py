from types import SimpleNamespace
from multiagent.spatial_program.object_graph import ObjectGraph

def obj(i,x,kind='building'):
    p=lambda x,y:SimpleNamespace(x=x,y=y)
    return SimpleNamespace(id=i,name='',object_type=kind,position=p(x,0),contour=[p(x-1,-1),p(x+1,-1),p(x+1,1)],area=2)
def test_nearby_and_type_filter_do_not_force_nearest_object():
    graph=ObjectGraph('map',[obj(1,3),obj(2,5),obj(3,4,'road')])
    assert [n.id for n,_ in graph.nearby([0,0],10,'building')]==[1,2]
