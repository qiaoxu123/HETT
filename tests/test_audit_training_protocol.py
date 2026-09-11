from pathlib import Path

from scripts.audit_training_protocol import WORKTREES, normalized, shell_options


def test_protocol_manifest_covers_every_experiment_worktree():
    assert WORKTREES == (
        "01-teacher-fix", "02-recovery", "03-grounding", "04-combined",
        "05-hypotheses", "06-bidir", "07-loss-ablation",
    )


def test_shell_options_reads_only_executable_command(tmp_path: Path):
    path = tmp_path / "train.sh"
    path.write_text(
        "python main.py --world_size 1 --seed 0 --altitude 50 "
        "--learning_rate 1e-4 --batch_size 2 --grad_accum 4 --epochs 20 "
        "--move_iteration 10 --max_action_len 20 --grid_size 5 "
        "--optim adamW --feedback student --train_trajectory_type mturk \"$@\"\n"
        "# example --epochs 1\n"
    )
    result = normalized(shell_options(path))
    assert result["epochs"] == 20
    assert result["batch_size"] * result["grad_accum"] == 8
    assert result["learning_rate"] == 1e-4
