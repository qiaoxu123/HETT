# Changelog

## [Unreleased] - 2026-09-09

### Features
- Added bidirectional cross-attention between target tokens and action/progress tokens.
- Replaced fixed multimodal token offsets with boundaries derived from runtime token lengths.

### Design Rationale
- Target prediction can now use action/progress context, while action and progress prediction can attend to map/goal and candidate target information.
- Target and motion updates are computed in parallel to avoid directional update-order bias.

### Notes & Caveats
- The target sequence consists of one map/goal token followed by all candidate-grid tokens.
- The motion sequence consists of one action token followed by one progress token.
- Runtime tests were not executed at the user's request.
