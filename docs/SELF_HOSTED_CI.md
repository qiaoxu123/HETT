# HETT self-hosted CI on rental-5001

Repository runner: **hett-rental-5001** (`self-hosted`, `Linux`, `X64`,
`hett`, `gpu`, `cuda`, `rental-5001`). One job at a time.

The runner is installed as the `20260922_1` user's systemd service
`github-actions-hett.service`. User lingering is enabled, so the service starts
at boot without an interactive login. It has restart-on-failure, lower CPU
priority, a 4-core CPU quota, and a 24 GiB RAM limit. No machine reboot was needed.

## Usage

GitHub → Actions → **HETT self-hosted CI** → Run workflow.
Select a branch in this repository (default: the visual diagnosis experiment).
The default job checks checkout, Python compilation and CUDA availability.
Enable `run_tests` to run the existing pytest suite on CPU. No package install,
model download, training, dataset scan or navigation rollout is implicit.
Logs and optional JUnit results are retained as Actions artifacts for 14 days.

```bash
gh workflow run self-hosted-ci.yml --repo qiaoxu123/HETT --ref main \
  -f ref=2027-CVPR/visual-overlap-geometry-diagnosis \
  -f run_tests=false -f gpu_check=true
```

Changes to Python sources or the workflow on `main` run the lightweight compile
check. Experiment branches without this workflow can still be checked through
manual dispatch from `main`. A `pull_request` trigger is intentionally absent.

## Installation and operations

- Runner: `/home/20260922_1/.local/share/github-actions/hett`
- Dedicated checkout/work: `/home/20260922_1/.local/share/github-actions/work/hett`
- Service: `/home/20260922_1/.config/systemd/user/github-actions-hett.service`
- Pre-job gate: `/home/20260922_1/.local/share/github-actions/hooks/hett-job-started.sh`
- Python runtime (`HETT_CI_PYTHON`): existing CityNav environment at
  `/home/rental/20260922_1/Workspace/DATA/rsrefseg2/venv/bin/python`.
  The workflow does not install or upgrade packages in this research environment.
- Initial runner release: `2.338.0`, official Linux x64 archive SHA256
  `af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32`.
  Runner-managed automatic updates remain enabled.

```bash
systemctl --user status github-actions-hett.service
systemctl --user restart github-actions-hett.service
journalctl --user -u github-actions-hett.service -n 100 --no-pager
systemctl --user disable --now github-actions-hett.service
```

Registration credentials are machine-local, outside the repository. To remove
the runner permanently, also remove its registration in GitHub Settings →
Actions → Runners.

## Execution boundary

HETT is public. External-fork workflow approval is set to
`all_external_contributors`. The machine's pre-job hook additionally rejects PR
and other events, other actors, and workflows outside the allowlisted HETT CI
workflow on `main` / the installation branch. This is a trusted-owner runner,
**not a sandbox for untrusted code**; approved code runs as the research user and
can access files that user can access. Do not add arbitrary PR triggers or run
unreviewed branch code here. GPU labels indicate hardware availability, not GPU
reservation; the lightweight default job does not train or allocate a model.

GitHub references:
[runner services](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/configure-the-application?platform=linux),
[pre-job hooks](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/run-scripts),
[self-hosted runner security](https://docs.github.com/en/actions/reference/security/secure-use).
