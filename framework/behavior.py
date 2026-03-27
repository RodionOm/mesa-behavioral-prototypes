class BehaviorMixin:
    """
    Minimal reusable behavior helper.

    This mixin separates:
    - behavior state initialization
    - decision selection
    - action execution
    - diagnostics updates
    - action history tracking

    It is intentionally lightweight:
    it supports action-oriented agents,
    but does not assume that all paradigms
    (e.g. BDI, RL) use the same decision mechanism.
    """

    def init_behavior_state(self, default_action="idle", history_length=8, action_names=None):
        self.last_action = default_action
        self.action_history = []
        self._behavior_history_length = history_length

        if action_names is None:
            action_names = []

        self.last_scores = {name: 0.0 for name in action_names}
        self.last_probabilities = {name: 0.0 for name in action_names}
        self._action_counts = {name: 0 for name in action_names}

    def decide(self):
        return self.model.policy.select_action(self)

    def act(self, action):
        action.execute(self)

    def update_behavior_history(self, action_name):
        self.action_history.append(action_name)

        if action_name in self._action_counts:
            self._action_counts[action_name] += 1

        if len(self.action_history) > self._behavior_history_length:
            self.action_history.pop(0)

    def behavioral_step(self):
        action, scores, probs = self.decide()

        self.last_action = action.name
        self.last_scores = scores
        self.last_probabilities = probs

        self.act(action)
        self.update_behavior_history(action.name)

    def get_action_persistence(self, action_name):
        return 0.05 if self.last_action == action_name else 0.0

    def get_action_fatigue(self, action_name):
        return 0.02 * self._action_counts.get(action_name, 0)

    def normalize_weights(self, attrs, floor=0.05):
        values = [max(getattr(self, attr), floor) for attr in attrs]
        total = sum(values)
        if total == 0:
            return

        for attr, value in zip(attrs, values):
            setattr(self, attr, value / total)


def build_action_diagnostics(action_names):
    return (
        {name: 0.0 for name in action_names},
        {name: 0.0 for name in action_names},
    )