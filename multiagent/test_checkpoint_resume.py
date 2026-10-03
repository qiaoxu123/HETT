import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from multiagent.agent import NavCMTAgent


def make_small_agent():
    agent = NavCMTAgent.__new__(NavCMTAgent)
    agent.args = SimpleNamespace(resume_optimizer=True)
    for name in ("lang_model", "vision_model", "vln_model"):
        model = torch.nn.Linear(2, 2)
        setattr(agent, f"{name}_without_ddp", model)
        optimizer_name = "et_optimizer" if name == "vln_model" else f"{name}_optimizer"
        setattr(agent, optimizer_name, torch.optim.Adam(model.parameters(), lr=1e-4))
    return agent


class CheckpointResumeTest(unittest.TestCase):
    def test_resumes_after_completed_epoch_with_optimizer_state(self):
        original = make_small_agent()
        for model, optimizer in (
            (original.lang_model_without_ddp, original.lang_model_optimizer),
            (original.vision_model_without_ddp, original.vision_model_optimizer),
            (original.vln_model_without_ddp, original.et_optimizer),
        ):
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "latest")
            original.save(1, checkpoint)
            resumed = make_small_agent()
            self.assertEqual(resumed.load(checkpoint), 2)
            for name in ("lang_model", "vision_model", "vln_model"):
                optimizer_name = "et_optimizer" if name == "vln_model" else f"{name}_optimizer"
                self.assertTrue(getattr(resumed, optimizer_name).state)
                torch.testing.assert_close(
                    getattr(resumed, f"{name}_without_ddp").weight,
                    getattr(original, f"{name}_without_ddp").weight,
                )


if __name__ == "__main__":
    unittest.main()
