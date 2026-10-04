import numpy as np
from multiagent.spatial_program.ordinal import ordinal_scores

def test_second_is_selected_along_resolved_axis():
    score,diag=ordinal_scores([[30,0],[10,0],[20,0]],[.1,.9,.5],[1,0],[0,0],2)
    assert diag['selected']==2 and int(np.argmax(score))==2
