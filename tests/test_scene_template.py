from multiagent.scene_grounding.scene_template import parse_scene_template


def test_scene_template_keeps_target_anchor_context_geometry_separate():
    template = parse_scene_template(
        "Fly past the church and stop at the large white building beside the parking lot.",
        target_phrase="large white building",
        landmark_names=("St Mary's Church",),
        surroundings=("parking lot",),
        target_object_type="Building",
    )
    assert template.target["type"] == "building"
    assert template.target["visual_attributes"]["color"] == ["white"]
    assert template.target["visual_attributes"]["size"] == ["large"]
    assert template.anchors[0]["name"] == "St Mary's Church"
    assert any(item["type"] == "parking_lot" for item in template.context)
    assert any(item["type"] == "past" for item in template.geometry)


def test_parser_is_deterministic():
    kwargs = dict(target_phrase="small red house", landmark_names=("Main Road",), target_object_type="Building")
    assert parse_scene_template("The small red house near Main Road", **kwargs) == parse_scene_template("The small red house near Main Road", **kwargs)
