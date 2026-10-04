from multiagent.geometry_reasoning.parser import parse_geometry
def test_parser():
 p=parse_geometry('Take the second building on the right past St Mary Church.',('St Mary Church',));types={x['type'] for x in p.constraints};assert {'ordinal','right_of','past'}<=types;assert p.anchors[0]['name']=='St Mary Church'
def test_annotation_not_in_text_is_rejected():assert not parse_geometry('Go to the building.',('Secret Church',)).anchors
