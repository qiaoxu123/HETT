def should_replan_stage2(stage1_ended, anchor_goal, predicted_goal, threshold_m):
    return bool(
        stage1_ended
        and anchor_goal is not None
        and predicted_goal.dist_to(anchor_goal) > threshold_m
    )


def should_stop_navigation(stage1_ended, progress, goal_distance_m,
                           stable_steps, args):
    return bool(
        stage1_ended
        and progress > args.progress_stop_threshold
        and goal_distance_m <= args.stop_goal_distance_m
        and stable_steps >= args.goal_stability_steps
    )
