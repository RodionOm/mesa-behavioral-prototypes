from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import CellAgent
from mesa.discrete_space.grid import OrthogonalMooreGrid

from framework.behavior import BehaviorMixin
from framework.policy import SoftmaxPolicy
from models.behavioral_information.actions import (
    ShareAction,
    VerifyAction,
    IgnoreAction,
    ExploreAction,
)


class InfoAgent(CellAgent, BehaviorMixin):
    def __init__(self, model, cell=None):
        super().__init__(model)
        if cell is not None:
            self.cell = cell

        self.curiosity = self.random.uniform(0.25, 0.80)
        self.confidence = self.random.uniform(0.20, 0.75)
        self.social_pressure = self.random.uniform(0.10, 0.60)
        self.topic_salience = self.random.uniform(0.20, 0.70)

        self.curiosity_weight = model.curiosity_weight
        self.confidence_weight = model.confidence_weight
        self.social_weight = model.social_weight

        self.current_regime = "neutral"

        self.init_behavior_state(
            default_action="ignore",
            history_length=model.history_length,
            action_names=["share", "verify", "ignore", "explore"],
        )

    def get_neighbor_cells(self):
        return list(self.cell.neighborhood)

    def get_neighbor_agents(self):
        neighbors = []
        for c in self.get_neighbor_cells():
            neighbors.extend(list(c.agents))
        return neighbors

    def average_neighbor_pressure(self):
        neighbors = self.get_neighbor_agents()
        if not neighbors:
            return 0.0
        return sum(n.social_pressure for n in neighbors) / len(neighbors)

    def average_neighbor_salience(self):
        neighbors = self.get_neighbor_agents()
        if not neighbors:
            return 0.0
        return sum(n.topic_salience for n in neighbors) / len(neighbors)

    def get_action_cost(self, action_name):
        if action_name == "verify":
            return 0.03 * (1.0 - self.curiosity)
        if action_name == "share":
            return 0.02 * (1.0 - self.confidence)
        return 0.0

    def update_regime(self):
        if self.social_pressure > 0.65 and self.confidence > 0.55:
            self.current_regime = "broadcast"
        elif self.confidence < 0.35:
            self.current_regime = "cautious"
        elif self.curiosity < 0.25 and self.topic_salience < 0.30:
            self.current_regime = "passive"
        else:
            self.current_regime = "neutral"

    def adapt_weights(self):
        if self.current_regime == "broadcast":
            self.social_weight += 0.18
        elif self.current_regime == "cautious":
            self.confidence_weight += 0.16
        elif self.current_regime == "passive":
            self.curiosity_weight = max(0.05, self.curiosity_weight - 0.05)

        self.normalize_weights(
            ["curiosity_weight", "confidence_weight", "social_weight"],
            floor=0.05,
        )

    def update_internal(self):
        neighbor_pressure = self.average_neighbor_pressure()
        neighbor_salience = self.average_neighbor_salience()

        self.social_pressure = min(1.0, 0.90 * self.social_pressure + 0.14 * neighbor_pressure)
        self.topic_salience = min(1.0, 0.88 * self.topic_salience + 0.15 * neighbor_salience)

        self.curiosity = max(0.0, min(1.0, self.curiosity + self.model.curiosity_drift))
        self.confidence = max(0.0, min(1.0, self.confidence + self.model.confidence_drift))

    def step(self):
        self.update_internal()
        self.update_regime()
        self.adapt_weights()

        if self.confidence < self.model.verify_interrupt_threshold:
            scores = self.model.policy.evaluate(self)
            probs = {k: 0.0 for k in scores}
            probs["verify"] = 1.0
            action = next(a for a in self.model.actions if a.name == "verify")

            self.last_action = action.name
            self.last_scores = scores
            self.last_probabilities = probs

            self.act(action)
            self.update_behavior_history(action.name)
        else:
            self.behavioral_step()


class InformationBehaviorModel(Model):
    def __init__(
        self,
        width=15,
        height=15,
        n_agents=50,
        curiosity_weight=1.0,
        confidence_weight=1.0,
        social_weight=1.0,
        curiosity_drift=0.01,
        confidence_drift=-0.003,
        memory_penalty_weight=0.15,
        temperature=0.14,
        score_scale=5.0,
        history_length=8,
        verify_interrupt_threshold=0.22,
        rng=None,
    ):
        super().__init__(rng=rng)

        self.width = width
        self.height = height

        self.curiosity_weight = curiosity_weight
        self.confidence_weight = confidence_weight
        self.social_weight = social_weight

        self.curiosity_drift = curiosity_drift
        self.confidence_drift = confidence_drift
        self.memory_penalty_weight = memory_penalty_weight
        self.history_length = history_length
        self.verify_interrupt_threshold = verify_interrupt_threshold

        self.grid = OrthogonalMooreGrid(
            (width, height),
            torus=True,
            capacity=10,
            random=self.random,
        )

        self.actions = [
            ShareAction(),
            VerifyAction(),
            IgnoreAction(),
            ExploreAction(),
        ]
        self.policy = SoftmaxPolicy(
            self.actions,
            temperature=temperature,
            score_scale=score_scale,
        )

        self.total_shares = 0
        self.total_verifications = 0

        all_cells = [self.grid[(x, y)] for x in range(width) for y in range(height)]
        chosen_cells = self.random.choices(all_cells, k=n_agents)

        for cell in chosen_cells:
            agent = InfoAgent(self, cell=cell)
            self.agents.add(agent)

        self.datacollector = DataCollector(
            model_reporters={
                "Sharing": lambda m: sum(a.last_action == "share" for a in m.agents),
                "Verifying": lambda m: sum(a.last_action == "verify" for a in m.agents),
                "Ignoring": lambda m: sum(a.last_action == "ignore" for a in m.agents),
                "Exploring": lambda m: sum(a.last_action == "explore" for a in m.agents),

                "Avg Curiosity": lambda m: sum(a.curiosity for a in m.agents) / len(m.agents),
                "Avg Confidence": lambda m: sum(a.confidence for a in m.agents) / len(m.agents),
                "Avg Social Pressure": lambda m: sum(a.social_pressure for a in m.agents) / len(m.agents),
                "Avg Topic Salience": lambda m: sum(a.topic_salience for a in m.agents) / len(m.agents),

                "Avg Share Score": lambda m: sum(a.last_scores["share"] for a in m.agents) / len(m.agents),
                "Avg Verify Score": lambda m: sum(a.last_scores["verify"] for a in m.agents) / len(m.agents),
                "Avg Ignore Score": lambda m: sum(a.last_scores["ignore"] for a in m.agents) / len(m.agents),
                "Avg Explore Score": lambda m: sum(a.last_scores["explore"] for a in m.agents) / len(m.agents),

                "Avg Share Prob": lambda m: sum(a.last_probabilities["share"] for a in m.agents) / len(m.agents),
                "Avg Verify Prob": lambda m: sum(a.last_probabilities["verify"] for a in m.agents) / len(m.agents),
                "Avg Ignore Prob": lambda m: sum(a.last_probabilities["ignore"] for a in m.agents) / len(m.agents),
                "Avg Explore Prob": lambda m: sum(a.last_probabilities["explore"] for a in m.agents) / len(m.agents),

                "Broadcast Regime": lambda m: sum(a.current_regime == "broadcast" for a in m.agents),
                "Cautious Regime": lambda m: sum(a.current_regime == "cautious" for a in m.agents),
                "Passive Regime": lambda m: sum(a.current_regime == "passive" for a in m.agents),

                "Total Shares": lambda m: m.total_shares,
                "Total Verifications": lambda m: m.total_verifications,
            }
        )
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")
        self.policy.decay_temperature(decay=0.998, min_temperature=0.04)
        self.datacollector.collect(self)