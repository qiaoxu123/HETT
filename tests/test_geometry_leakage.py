from pathlib import Path
from multiagent.geometry_reasoning.parser import parse_geometry
def test_parser_has_no_goal_argument():assert 'goal' not in parse_geometry.__code__.co_varnames
def test_no_rgb_or_test_split():
 s=(Path(__file__).parents[1]/'scripts/evaluate_geometry_reasoner.py').read_text();assert 'import cv2' not in s;assert 'imageprocessor' not in s.casefold();assert "('train_seen','val_seen','val_unseen')" in s;assert 'citynav_test_unseen' not in s
def test_dynamic_pose_is_prefix_indexed():
 s=(Path(__file__).parents[1]/'scripts/evaluate_geometry_reasoner.py').read_text();assert "progress*(len(row['trajectory'])-1)" in s
