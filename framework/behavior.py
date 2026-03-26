class BehaviorMixin:
    """Reusable helper layer for behavior-driven agents."""

    def init_behavior_state(self, default_action="idle", history_length=8):
        self.last_action = default_action
        self.action_history = []
        self._behavior_history_length = history_length

    def get_action_persistence(self, action_name, strength=0.08):
        return strength if self.last_action == action_name else 0.0

    def get_action_fatigue(self, action_name):
        if not self.action_history:
            return 0.0
        ratio = self.action_history.count(action_name) / len(self.action_history)
        return self.model.memory_penalty_weight * ratio

    def get_action_cost(self, action_name):
        """Override in domain-specific agents when needed."""
        return 0.0

    def update_history(self, action_name):
        self.action_history.append(action_name)
        if len(self.action_history) > self._behavior_history_length:
            self.action_history.pop(0)

    def normalize_weights(self, names, floor=0.05):
        total = 0.0
        for name in names:
            value = max(floor, getattr(self, name))
            setattr(self, name, value)
            total += value

        if total <= 0:
            equal = 1.0 / len(names)
            for name in names:
                setattr(self, name, equal)
            return

        for name in names:
            setattr(self, name, getattr(self, name) / total)