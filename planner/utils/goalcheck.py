"""Provide a class for easy checking if a goal state is reached."""
import copy


class GoalReachedChecker:
    """GoalChecker for easy checking if the goal is reached."""

    def __init__(self, planning_problem):
        """__init__ function."""
        self.goal = planning_problem.goal
        self.status = []

    def register_current_state(self, current_state):
        """Register the current state and check if in goal."""
        self.status = []
        for goal_state in self.goal.state_list:
            state_status = {}
            normalized_state = self._normalize_states(current_state, goal_state)
            self._check_position(normalized_state, goal_state, state_status)
            self._check_orientation(normalized_state, goal_state, state_status)
            self._check_velocity(normalized_state, goal_state, state_status)
            self._check_time_step(normalized_state, goal_state, state_status)
            self.status.append(state_status)

    def goal_reached_status(self, ignore_exceeded_time=False):
        """Get the goal status."""
        for state_status in copy.deepcopy(self.status):
            if "time_step" in state_status:
                timing_flag = state_status.pop("timing_flag")
            if all(list(state_status.values())):
                return True
            elif "time_step" in state_status:
                _ = state_status.pop("time_step")
                if (
                    ignore_exceeded_time
                    and timing_flag == "exceeded"
                    and all(list(state_status.values()))
                ):
                    return True

            return False

    def _check_position(self, normalized_state, goal_state, state_status):
        if hasattr(goal_state, "position"):
            state_status["position"] = goal_state.position.contains_point(
                normalized_state.position
            )

    def _check_orientation(self, normalized_state, goal_state, state_status):
        if hasattr(goal_state, "orientation"):
            state_status["orientation"] = self.goal._check_value_in_interval(
                normalized_state.orientation, goal_state.orientation
            )

    def _check_velocity(self, normalized_state, goal_state, state_status):
        if hasattr(goal_state, "velocity"):
            state_status["velocity"] = self.goal._check_value_in_interval(
                normalized_state.velocity, goal_state.velocity
            )

    def _check_time_step(self, normalized_state, goal_state, state_status):
        if hasattr(goal_state, "time_step"):
            state_status["time_step"] = self.goal._check_value_in_interval(
                normalized_state.time_step, goal_state.time_step
            )
            if normalized_state.time_step > goal_state.time_step.end:
                state_status["timing_flag"] = "exceeded"
            else:
                state_status["timing_flag"] = "not exceeded"

    def _normalize_states(self, current_state, goal_state):
        goal_state_tmp = copy.deepcopy(goal_state)
        goal_state_fields = {
            slot for slot in goal_state.__slots__ if hasattr(goal_state, slot)
        }
        state_fields = {
            slot for slot in goal_state.__slots__ if hasattr(current_state, slot)
        }
        (
            state_new,
            state_fields,
            goal_state_tmp,
            goal_state_fields,
        ) = self.goal._harmonize_state_types(
            current_state, goal_state_tmp, state_fields, goal_state_fields
        )

        if not goal_state_fields.issubset(state_fields):
            raise ValueError(
                "The goal states {} are not a subset of the provided states {}!".format(
                    goal_state_fields, state_fields
                )
            )

        return state_new
