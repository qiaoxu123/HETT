"""Unit tests for the 3D-anchored multi-view entity-grounding round.

Two things have to be true before any number from this round means anything:
that the world-coordinate correspondence between the two views is the real
geometry rather than a plausible-looking mask, and that nothing in the pipeline
quietly special-cases one kind of entity.  The first is pinned here by
constructing a correspondence analytically and checking that the coherence
measure separates it from a shuffle; the second by exercising the support ladder
and the scale policy on entities of very different sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sensaturban_fpv.entity_geometry import (  # noqa: E402
    CLASS_IDS, EntityView, CropWindow, ScaleView, box_membership,
    context_extent_m, correspondence_coherence, correspondence_matrix,
    entity_bounds, entity_scales, group_of, polygon_membership,
    shuffled_correspondence, support_margin, tight_extent_m,
)


# --------------------------------------------------------------------------
# support geometry
# --------------------------------------------------------------------------

def test_box_membership_is_inclusive_at_the_faces():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.01, 0.5, 0.5],
                    [0.5, -0.01, 0.5]])
    assert box_membership(pts, lo, hi).tolist() == [True, True, False, False]


def test_polygon_membership_matches_a_hand_counted_square():
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    pts = np.array([[1.0, 1.0], [2.5, 1.0], [-0.5, 1.0], [0.0, 2.0], [2.0, 0.0]])
    inside = polygon_membership(pts, square)
    assert inside[0] and not inside[1] and not inside[2]
    # The two vertices lie exactly on the boundary.  The half-open rule on y
    # excludes both, which is what keeps a shared vertex from being counted
    # twice when two objects tile a row -- the direction of the exclusion is
    # pinned here so a change to it cannot pass silently.
    assert not inside[3] and not inside[4]


def test_polygon_membership_handles_a_concave_polygon():
    # An L shape: the notch must read as outside.
    poly = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0],
                     [1.0, 3.0], [0.0, 3.0]])
    pts = np.array([[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]])
    assert polygon_membership(pts, poly).tolist() == [True, True, False, True]


def test_polygon_membership_without_a_polygon_keeps_everything():
    pts = np.array([[5.0, 5.0], [-3.0, 0.0]])
    assert polygon_membership(pts, None).all()
    assert polygon_membership(pts, np.zeros((0, 2))).all()


def test_support_margin_scales_with_the_entity():
    car = support_margin((4.5, 1.8, 1.5))
    warehouse = support_margin((36.0, 80.0, 11.0))
    bucket = support_margin((0.4, 0.4, 0.6))
    assert bucket == pytest.approx(0.1)
    assert 0.1 < car < 1.0
    assert warehouse == pytest.approx(1.0)
    # The point of the scaling: a car's box must not reach the next car.
    assert car < 0.5


def test_entity_scales_never_crop_the_entity_out_of_its_own_tight_view():
    for dim in [(4.5, 1.8, 1.5), (15.0, 8.0, 10.0), (36.7, 81.6, 10.9),
                (130.0, 40.0, 20.0), (1.6, 1.6, 2.5)]:
        scales = entity_scales(dim)
        assert scales["tight"] >= max(dim[0], dim[1])
        assert scales["context"] >= scales["tight"] - 1e-9
        assert scales["tight"] >= 6.0 - 1e-9


def test_tight_scale_is_strictly_smaller_for_a_normal_building():
    scales = entity_scales((20.0, 30.0, 12.0))
    assert scales["tight"] < scales["context"]


def test_extent_helpers_agree_with_the_scale_policy():
    dim = (10.0, 20.0, 5.0)
    assert tight_extent_m(dim) == entity_scales(dim)["tight"]
    assert context_extent_m(dim) == entity_scales(dim)["context"]


def test_entity_bounds_is_centred_on_the_annotation():
    lo, hi = entity_bounds([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], margin=0.5)
    assert lo.tolist() == [-0.5, -0.5, -0.5]
    assert hi.tolist() == [2.5, 4.5, 6.5]


def test_type_groups_cover_every_class_the_dataset_names():
    for name in CLASS_IDS:
        assert group_of(name) in ("building", "vehicle", "other")
    assert group_of("Building") == "building"
    assert group_of("Car") == "vehicle"
    assert group_of("Bike") == "vehicle"
    assert group_of("Wall") == "other"
    # An unknown type must not raise: it simply is not a building or a vehicle.
    assert group_of("Something New") == "other"


# --------------------------------------------------------------------------
# crop windows
# --------------------------------------------------------------------------

def test_patch_of_uv_maps_the_crop_uniformly():
    win = CropWindow(centre=(100.0, 100.0), side=64.0, patches=32)
    rows, cols, ok = win.patch_of_uv(np.array([100.0, 68.0, 131.0]),
                                     np.array([100.0, 68.0, 131.0]))
    assert ok.tolist() == [True, True, True]
    assert rows.tolist() == [16, 0, 31]
    assert cols.tolist() == [16, 0, 31]


def test_patch_of_uv_rejects_pixels_outside_the_crop():
    win = CropWindow(centre=(0.0, 0.0), side=10.0, patches=8)
    _, _, ok = win.patch_of_uv(np.array([-5.01, 4.99]), np.array([0.0, 0.0]))
    assert ok.tolist() == [False, True]


# --------------------------------------------------------------------------
# correspondence
# --------------------------------------------------------------------------

def _view_with_correspondence(td, ob, weight, patch_grid=8):
    scales = {"td": {}, "oblique": {}}
    mask_td = np.zeros((patch_grid, patch_grid), dtype=bool)
    mask_td.ravel()[np.unique(td)] = True
    mask_o = np.zeros((patch_grid, patch_grid), dtype=bool)
    mask_o.ravel()[np.unique(ob)] = True
    for view, mask in (("td", mask_td), ("oblique", mask_o)):
        scales[view]["context"] = ScaleView(
            name="context", extent_m=10.0, window=None, patch_mask=mask,
            patch_count=int(mask.sum()), visible_points=1, projected_pixels=1)
    return EntityView(entity_id=1, object_type="Car", support_source="box",
                      support_points=1, scales=scales,
                      correspondence={"td": np.asarray(td, np.int64),
                                      "o": np.asarray(ob, np.int64),
                                      "weight": np.asarray(weight, np.float32)})


def test_correspondence_matrix_is_row_normalised():
    view = _view_with_correspondence([0, 0, 9], [0, 1, 9], [1.0, 3.0, 2.0])
    W = correspondence_matrix(view, 8, symmetric=False)
    assert W[0].sum() == pytest.approx(1.0)
    assert W[0, 0] == pytest.approx(0.25)
    assert W[0, 1] == pytest.approx(0.75)
    assert W[9, 9] == pytest.approx(1.0)
    assert W[5].sum() == pytest.approx(0.0)


def test_correspondence_coherence_prefers_a_smooth_map_to_a_shuffled_one():
    # A real alignment moves smoothly across the patch grid; the shuffle keeps
    # every weight and every visited patch but destroys that.
    grid = 8
    td = list(range(0, 16))
    # rows 0 and 1 of the grid map to rows 2 and 3 of the oblique grid
    ob = [2 * grid + (p % grid) for p in td]
    view = _view_with_correspondence(td, ob, np.ones(len(td)))
    shuffled = shuffled_correspondence(view, grid, seed=3)
    assert correspondence_coherence(view, grid) < correspondence_coherence(shuffled, grid)


def test_shuffle_preserves_weights_and_the_patch_pool():
    grid = 8
    td = [0, 1, 2, 3, 8, 9, 10, 11]
    ob = [16, 17, 18, 19, 24, 25, 26, 27]
    weight = np.arange(1, 9, dtype=np.float32)
    view = _view_with_correspondence(td, ob, weight)
    shuffled = shuffled_correspondence(view, grid, seed=11)
    assert np.array_equal(shuffled.correspondence["weight"],
                          view.correspondence["weight"])
    assert np.array_equal(shuffled.correspondence["td"], view.correspondence["td"])
    assert sorted(shuffled.correspondence["o"].tolist()) == sorted(ob)
    mask = view.scales["oblique"]["context"].patch_mask.ravel()
    assert mask[shuffled.correspondence["o"]].all()


def test_shuffle_is_deterministic_for_a_seed():
    view = _view_with_correspondence(list(range(9)), list(range(9)),
                                     np.ones(9, np.float32))
    a = shuffled_correspondence(view, 8, seed=5)
    b = shuffled_correspondence(view, 8, seed=5)
    c = shuffled_correspondence(view, 8, seed=6)
    assert np.array_equal(a.correspondence["o"], b.correspondence["o"])
    assert not np.array_equal(a.correspondence["o"], c.correspondence["o"])


# --------------------------------------------------------------------------
# missing scales
# --------------------------------------------------------------------------

def test_accessors_return_an_empty_mask_for_a_view_with_no_scale():
    view = EntityView(entity_id=1, object_type="Car", support_source="box",
                      support_points=0, scales={"td": {}, "oblique": {}})
    mask = view.mask_of("oblique", "context", 8)
    assert mask.shape == (8, 8) and not mask.any()
    assert view.count_of("oblique", "context") == 0
    assert view.window_of("oblique", "context") is None


def test_a_zero_mask_never_matches_shuffle_requirements():
    view = EntityView(entity_id=1, object_type="Car", support_source="box",
                      support_points=0, scales={"td": {}, "oblique": {}},
                      correspondence={"td": np.array([0]), "o": np.array([1]),
                                      "weight": np.array([1.0], np.float32)})
    assert shuffled_correspondence(view, 8, seed=1) is view


# --------------------------------------------------------------------------
# point-cloud index additions
# --------------------------------------------------------------------------

def test_query_box_slots_label_filter_runs_before_the_cap():
    from sensaturban_fpv.plyio import XyGridIndex

    class _Cloud:
        path = Path("/tmp/does-not-matter.ply")

        def __len__(self):
            return 6

    grid = XyGridIndex(_Cloud(), cell=1.0)
    grid.lo = np.array([0.0, 0.0])
    grid.hi = np.array([10.0, 10.0])
    grid.nx, grid.ny = 10, 10
    grid.starts = np.array([0, 6] + [6] * 98)

    # All six points share one bucket; only the last three carry label 9.
    xyz = np.zeros((6, 3), dtype=np.float32)
    xyz[:, 0] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    labels = np.array([0, 0, 0, 9, 9, 9], dtype=np.uint8)
    grid.order = np.arange(6, dtype=np.int32)

    class _Arrays:
        shape = xyz.shape
        dtype = xyz.dtype

        def __getitem__(self, item):
            return xyz[item]

    pts = grid.query_box_slots(np.array([0.0, -1.0, -1.0]),
                               np.array([5.0, 1.0, 1.0]), _Arrays(),
                               max_points=2, label_sets=(labels, np.array([9])))
    # The label filter ran first, so the cap decimated three car points rather
    # than six mixed ones; without it the cap would have returned road points.
    assert 0 < len(pts) <= 2
    np.testing.assert_allclose(sorted(pts[:, 0].tolist()), [1.3, 1.5], atol=1e-6)
    unfiltered = grid.query_box_slots(np.array([0.0, -1.0, -1.0]),
                                      np.array([5.0, 1.0, 1.0]), _Arrays(),
                                      max_points=2)
    np.testing.assert_allclose(sorted(unfiltered[:, 0].tolist()), [1.0, 1.3],
                               atol=1e-6)


# --------------------------------------------------------------------------
# the fusion model
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch")


class _StubHead(torch.nn.Module):
    """The shape of SigLIP's pooling head: probe, attention, layernorm, mlp."""

    def __init__(self, dim=16, num_heads=2):
        super().__init__()
        self.probe = torch.nn.Parameter(torch.randn(1, 1, dim) * 0.1)
        self.attention = torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.layernorm = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, 2 * dim), torch.nn.GELU(),
            torch.nn.Linear(2 * dim, dim))


class _StubVision(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.vision_model = torch.nn.Module()
        self.vision_model.head = _StubHead(dim)


def _synthetic_sample(dim=16, K=6, n_td=24, n_o=30, n_corr=40):
    rng = np.random.default_rng(0)
    td_row = np.sort(rng.integers(0, K, n_td))
    o_row = np.sort(rng.integers(0, K, n_o))
    corr_cand = np.sort(rng.integers(0, K, n_corr))
    corr_td = rng.integers(0, n_td, n_corr)
    corr_o = rng.integers(0, n_o, n_corr)
    arrays = {
        "td_tokens": rng.standard_normal((n_td, dim)).astype(np.float16),
        "oblique_tokens": rng.standard_normal((n_o, dim)).astype(np.float16),
        "td_row": td_row.astype(np.int32),
        "oblique_row": o_row.astype(np.int32),
        "corr_cand": corr_cand.astype(np.int32),
        "corr_td_row": corr_td.astype(np.int32),
        "corr_o_row": corr_o.astype(np.int32),
        "corr_w": rng.random(n_corr).astype(np.float32),
        "geometry": rng.standard_normal((K, 19)).astype(np.float32),
        "entity_ids": np.arange(K),
    }
    for name in ("td_tight_masked", "oblique_tight_masked", "td_context_masked",
                 "oblique_context_masked", "td_context_background",
                 "oblique_context_background", "td_context_global",
                 "oblique_context_global", "td_exclusive", "oblique_exclusive"):
        arrays[name] = rng.standard_normal((K, dim)).astype(np.float32)
    for name in ("td_context_masked", "td_tight_masked", "td_context_background",
                 "td_context_global", "td_exclusive"):
        arrays[name] /= np.linalg.norm(arrays[name], axis=1, keepdims=True)

    class _Sample:
        pass

    sample = _Sample()
    sample.arrays = arrays
    sample.key = "synthetic"
    return sample, dim, K


def _build(components, fusion, shared, dim):
    from train_entity_grounding import Inputs, make_model

    module = make_model(torch, _StubVision(dim), dim, components, fusion, shared, 19,
                        np.zeros(19, np.float32), np.ones(19, np.float32))
    return module, Inputs


def test_entity_model_scores_every_candidate_and_stays_under_the_parameter_budget():
    from train_entity_grounding import APPEARANCE, EXCLUSIVE, SURROUNDINGS

    sample, dim, K = _synthetic_sample()
    components = APPEARANCE + SURROUNDINGS + EXCLUSIVE + ("geometry",)
    module, Inputs = _build(components, "attn", True, dim)
    n_params = sum(p.numel() for p in module.parameters())
    assert n_params < 1_000_000
    inputs = Inputs(torch, sample, "cpu")
    arrays = {k: torch.from_numpy(sample.arrays[k]) for k in components}
    arrays["geometry"] = torch.from_numpy(sample.arrays["geometry"])
    text = torch.zeros(dim)
    text[0] = 1.0
    scores, reverse, joint = module(arrays, text, inputs)
    assert scores.shape == (K,)
    assert joint.shape == (K, dim)
    assert torch.allclose(joint.norm(dim=-1), torch.ones(K), atol=1e-4)
    assert reverse is None  # only the XL variant asks for the reverse branch


def test_frozen_readout_parameters_are_not_in_the_optimizer_set():
    """The optimizer is built from ``module.parameters()``.

    Assigning an ``nn.Module`` to an attribute registers it as a submodule, so
    the frozen vision tower has to be kept out of the module tree on purpose --
    otherwise the first ``opt.step()`` would fine-tune SigLIP2.
    """
    from train_entity_grounding import APPEARANCE

    sample, dim, K = _synthetic_sample()
    module, _ = _build(APPEARANCE + ("geometry",), "concat", False, dim)
    names = {n for n, _ in module.named_parameters()}
    assert not any(n.startswith("vision") for n in names), names
    assert not any(n.startswith("vision") for n in dict(module.named_buffers()))
    assert module.vision is not None  # still reachable for the readout


def test_fusion_gradients_reach_the_pair_module():
    from train_entity_grounding import APPEARANCE, Inputs

    sample, dim, K = _synthetic_sample()
    module, _ = _build(APPEARANCE + ("geometry",), "attn", True, dim)
    inputs = Inputs(torch, sample, "cpu")
    arrays = {k: torch.from_numpy(sample.arrays[k]) for k in APPEARANCE}
    arrays["geometry"] = torch.from_numpy(sample.arrays["geometry"])
    text = torch.ones(dim) / np.sqrt(dim)
    target = torch.tensor([3])

    def step():
        module.zero_grad(set_to_none=True)
        scores, _, _ = module(arrays, text, inputs)
        torch.nn.functional.cross_entropy((10 * scores)[None, :], target).backward()

    # The last layer is zero-initialised, so on the very first backward pass it
    # is the only part of the pair module that can receive a gradient.  The
    # first layer gets one only once the last layer has moved.
    step()
    assert float(module.pair[-1].weight.grad.abs().sum()) > 0
    assert float(module.pair[0].weight.grad.abs().sum()) == 0
    assert float(module.enc[0].weight.grad.abs().sum()) > 0
    with torch.no_grad():
        module.pair[-1].weight.add_(module.pair[-1].weight.grad)
    step()
    assert float(module.pair[0].weight.grad.abs().sum()) > 0


def test_shuffled_correspondence_changes_the_score_but_not_the_patch_pool():
    from train_entity_grounding import APPEARANCE, Inputs

    sample, dim, K = _synthetic_sample()
    module, _ = _build(APPEARANCE + ("geometry",), "attn", True, dim)
    module.eval()
    arrays = {k: torch.from_numpy(sample.arrays[k]) for k in APPEARANCE}
    arrays["geometry"] = torch.from_numpy(sample.arrays["geometry"])
    text = torch.ones(dim) / np.sqrt(dim)
    with torch.no_grad():
        straight = module(arrays, text, Inputs(torch, sample, "cpu"))[0]
        shuffled = module(arrays, text, Inputs(torch, sample, "cpu",
                                               shuffle_seed=7))[0]
    assert straight.shape == shuffled.shape == (K,)
    assert not torch.allclose(straight, shuffled, atol=1e-6)
    assert set(np.unique(sample.arrays["corr_o_row"])) == set(np.unique(
        Inputs(torch, sample, "cpu", shuffle_seed=7).corr_o_row.numpy()))


def test_zero_initialised_residual_starts_at_the_view_mean():
    from train_entity_grounding import APPEARANCE, Inputs

    sample, dim, K = _synthetic_sample()
    module, _ = _build(APPEARANCE + ("geometry",), "concat", False, dim)
    module.eval()
    arrays = {k: torch.from_numpy(sample.arrays[k]) for k in APPEARANCE}
    arrays["geometry"] = torch.from_numpy(sample.arrays["geometry"])
    # With a zero-initialised second layer the concat head is exactly zero, so
    # the first step of training is the mean of the encoded views and nothing
    # else -- the ablation begins from the baseline it has to beat.
    with torch.no_grad():
        stacked = torch.stack([module.enc(arrays[k]) for k in APPEARANCE]
                              + [module.geom_mlp((arrays["geometry"]
                                                  - module.geom_mean)
                                                 / module.geom_std)], dim=1)
        delta = module.concat_head(stacked.flatten(1))
    assert torch.allclose(delta, torch.zeros_like(delta), atol=1e-6)
