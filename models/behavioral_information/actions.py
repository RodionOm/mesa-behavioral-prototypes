from framework.action import Action


class ShareAction(Action):
    name = "share"

    def score(self, agent):
        return max(
            0.0,
            0.45 * (agent.confidence ** 2)
            + 0.30 * agent.social_pressure
            + 0.15 * agent.topic_salience
            + agent.get_action_persistence(self.name)
            - agent.get_action_fatigue(self.name)
            - agent.get_action_cost(self.name),
        )

    def execute(self, agent):
        agent.confidence = min(1.0, agent.confidence + 0.06)
        agent.curiosity = max(0.0, agent.curiosity - 0.03)
        agent.social_pressure = max(0.0, agent.social_pressure - 0.05)
        agent.model.total_shares += 1


class VerifyAction(Action):
    name = "verify"

    def score(self, agent):
        uncertainty = 1.0 - agent.confidence
        return max(
            0.0,
            0.45 * agent.curiosity
            + 0.30 * (uncertainty ** 2)
            + 0.15 * agent.topic_salience
            + agent.get_action_persistence(self.name)
            - agent.get_action_fatigue(self.name)
            - agent.get_action_cost(self.name),
        )

    def execute(self, agent):
        agent.confidence = min(1.0, agent.confidence + 0.10)
        agent.curiosity = max(0.0, agent.curiosity - 0.02)
        agent.model.total_verifications += 1


class IgnoreAction(Action):
    name = "ignore"

    def score(self, agent):
        return max(
            0.0,
            0.35 * (1.0 - agent.curiosity)
            + 0.20 * (1.0 - agent.social_pressure)
            + 0.15 * (1.0 - agent.topic_salience)
            + agent.get_action_persistence(self.name)
            - agent.get_action_fatigue(self.name),
        )

    def execute(self, agent):
        agent.social_pressure = max(0.0, agent.social_pressure - 0.03)
        agent.topic_salience = max(0.0, agent.topic_salience - 0.05)
        agent.curiosity = max(0.0, agent.curiosity - 0.01)


class ExploreAction(Action):
    name = "explore"

    def score(self, agent):
        return max(
            0.0,
            0.40 * agent.curiosity
            + 0.20 * (1.0 - agent.confidence)
            + 0.10 * (1.0 - agent.social_pressure)
            + 0.15 * (1.0 - agent.topic_salience)
            + agent.get_action_persistence(self.name)
            - agent.get_action_fatigue(self.name),
        )

    def execute(self, agent):
        agent.curiosity = min(1.0, agent.curiosity + 0.05)
        agent.topic_salience = min(1.0, agent.topic_salience + 0.07)
        agent.social_pressure = min(1.0, agent.social_pressure + 0.02)