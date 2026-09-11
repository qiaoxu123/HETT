"""Small real-data GPU check; never writes training checkpoints."""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from torch.utils.data import DataLoader
from parser import parse_args
from agent import NavCMTAgent
from env import CityNavBatch
from types import SimpleNamespace


def check_accumulation():
    class Loader:
        dataset = SimpleNamespace(size=lambda: 5)
        def __iter__(self):
            return iter(range(5))
    agent = NavCMTAgent.__new__(NavCMTAgent)
    agent.args = SimpleNamespace(grad_accum=4, teacher_weight=1)
    agent.env = SimpleNamespace(batch_size=1)
    agent.default_gpu = False
    models = [torch.nn.Linear(1, 1, bias=False) for _ in range(3)]
    agent.lang_model, agent.vision_model, agent.vln_model = models
    for model in models:
        model.weight.data.zero_()
    agent.optimizers = tuple(torch.optim.SGD(m.parameters(), lr=0.1) for m in models)
    agent.lang_model_optimizer, agent.vision_model_optimizer, agent.et_optimizer = agent.optimizers
    values = iter([1., 2., 3., 4., 5.])
    def rollout(**unused):
        value = next(values)
        agent.loss = sum((m.weight-value).square().sum() for m in models)
    agent.rollout = rollout
    agent.train(Loader(), 1, feedback='teacher')
    for model in models:
        torch.testing.assert_close(model.weight, torch.tensor([[1.4]]))


def main():
    check_accumulation()
    args = parse_args()
    assert args.max_episodes > 0, 'Use --max_episodes for this quick check'
    torch.manual_seed(args.seed)
    env = CityNavBatch('train_seen', args, batch_size=args.batch_size)
    agent = NavCMTAgent(args)
    agent.env = env
    module = agent.vln_model.task_interaction
    before = {n: p.detach().clone() for n, p in module.named_parameters()}
    gradients = defaultdict(float)
    handles = [p.register_hook(lambda g, n=n: gradients.__setitem__(n, max(gradients[n], g.norm().item())))
               for n, p in module.named_parameters()]
    updates = []
    h = agent.et_optimizer.register_step_post_hook(lambda *unused: updates.append(1))
    torch.cuda.reset_peak_memory_stats()
    agent.train(DataLoader(env, batch_size=1), 1, feedback='student')
    expected = ((env.size() + args.batch_size - 1) // args.batch_size + args.grad_accum - 1) // args.grad_accum
    assert len(updates) == expected, (len(updates), expected)
    for n, p in module.named_parameters():
        assert gradients[n] > 0 and torch.isfinite(p).all(), n
        assert not torch.equal(before[n], p), n
    for handle in handles:
        handle.remove()
    h.remove()
    assert all(torch.isfinite(torch.tensor(agent.logs[k])).all() for k in ['IL_loss', 'direction_loss'])

    # Both directions must depend on the opposite input, independently.
    module.eval()
    target = torch.randn(2, 26, args.demb, device='cuda', requires_grad=True)
    motion = torch.randn(2, 2, args.demb, device='cuda', requires_grad=True)
    out_t, out_m = module(target, motion)
    tm = torch.autograd.grad(out_t.square().mean(), motion, retain_graph=True)[0].norm().item()
    mt = torch.autograd.grad(out_m.square().mean(), target)[0].norm().item()
    assert tm > 0 and mt > 0

    # Same real observation, interaction on/off: all four task heads must change.
    agent.vln_model.eval()
    agent.lang_model.eval()
    agent.vision_model.eval()
    captured = {}
    def capture(model, positional, kwargs):
        if not captured:
            captured.update({k: v.detach().clone() if torch.is_tensor(v) else v for k, v in kwargs.items()})
    handle = agent.vln_model.register_forward_pre_hook(capture, with_kwargs=True)
    agent.test(DataLoader(env, batch_size=1))
    handle.remove()
    with torch.no_grad():
        enabled = agent.vln_model(**captured)
        args.disable_task_interaction = True
        disabled = agent.vln_model(**captured)
        args.disable_task_interaction = False
        # Dynamic modality lengths, including more than one direction/map token.
        encoder = agent.vln_model.encoder_vl
        inputs = [torch.randn(2, length, args.demb, device='cuda') for length in [4, 3, 2, 2, 25]]
        mask = torch.tensor([[1,1,0,0], [1,1,1,1]], device='cuda')
        # Positional encoding mutates modality embeddings; use independent copies.
        changed = [x.clone() for x in inputs]
        changed[0][0, 2:4] += 100
        fused, padding = encoder.forward_with_map(*inputs, lang_mask=mask)
        assert fused.shape == (2, 36, args.demb) and padding[0, 2:4].all()
        fused_changed, _ = encoder.forward_with_map(*changed, lang_mask=mask)
        torch.testing.assert_close(fused[0, 4:], fused_changed[0, 4:])
    deltas = [(a-b).abs().max().item() for a,b in zip(enabled[:4], disabled[:4])]
    assert all(d > 0 for d in deltas)
    result = dict(device=torch.cuda.get_device_name(), torch=torch.__version__, episodes=env.size(),
                  optimizer_steps=len(updates), interaction_parameters_with_grad=len(gradients),
                  cross_gradients=[tm, mt], head_output_deltas=deltas,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  losses=agent.logs['IL_loss'])
    print('\nGPU_CHECK ' + json.dumps(result))


if __name__ == '__main__':
    main()
