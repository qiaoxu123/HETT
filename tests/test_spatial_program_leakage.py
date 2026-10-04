import inspect
from multiagent.spatial_program import parser,executor

def test_parser_executor_have_no_gt_or_rgb_interface():
    text=inspect.getsource(parser)+inspect.getsource(executor)
    for forbidden in ('target_positions','goal_coordinate','future_pose','test_unseen','rgb'):
        assert forbidden not in text.casefold()
