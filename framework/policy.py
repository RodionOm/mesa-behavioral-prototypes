import math


class SoftmaxPolicy:
    """
    Reusable stochastic policy for action selection.

    This policy is intentionally generic enough for action-oriented agents,
    but it does not assume that all behavioral paradigms should use softmax.
    """

    def __init__(self, actions, temperature=0.12, score_scale=6.0, entropy_bonus=0.02):
        self.actions = actions
        self.temperature = temperature
        self.score_scale = score_scale
        self.entropy_bonus = entropy_bonus

    def evaluate(self, agent):
        scores = {a.name: a.score(agent) for a in self.actions}
        for key in scores:
            scores[key] += self.entropy_bonus
        return scores

    def softmax(self, scores):
        temp = max(self.temperature, 1e-6)
        scaled = {k: v * self.score_scale for k, v in scores.items()}
        max_score = max(scaled.values())

        exp_scores = {
            k: math.exp((v - max_score) / temp)
            for k, v in scaled.items()
        }
        total = sum(exp_scores.values())

        if total == 0:
            n = len(exp_scores)
            return {k: 1.0 / n for k in exp_scores}

        return {k: v / total for k, v in exp_scores.items()}

    def select_action(self, agent):
        scores = self.evaluate(agent)
        probs = self.softmax(scores)

        names = list(probs.keys())
        weights = list(probs.values())
        chosen_name = agent.random.choices(names, weights=weights, k=1)[0]
        chosen_action = next(a for a in self.actions if a.name == chosen_name)

        return chosen_action, scores, probs

    def decay_temperature(self, decay=0.997, min_temperature=0.03):
        self.temperature = max(min_temperature, self.temperature * decay)