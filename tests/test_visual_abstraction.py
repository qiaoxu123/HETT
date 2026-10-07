import numpy as np
import pytest

from multiagent.visual_goal.abstraction import abstract, information_table


def test_levels_preserve_shape_and_are_deterministic():
    image=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
    regions={"target":np.zeros((32,32),np.uint8),"anchor":np.zeros((32,32),np.uint8)}
    regions["target"][10:15,10:15]=255; regions["anchor"][20:24,20:24]=255
    for level in (f"L{i}" for i in range(10)):
        result=abstract(image,level,regions)
        assert result.shape==image.shape and result.dtype==np.uint8
        assert np.array_equal(result,abstract(image,level,regions))


def test_unknown_level_rejected():
    with pytest.raises(ValueError): abstract(np.zeros((8,8,3),np.uint8),"L10")


def test_information_table_covers_all_levels():
    assert [r["level"] for r in information_table()]==[f"L{i}" for i in range(10)]
