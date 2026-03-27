from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import CellAgent
from mesa.discrete_space.grid import OrthogonalMooreGrid

from framework.behavior import BehaviorMixin
from framework.policy import SoftmaxPolicy
from models.behavioral_survival.actions import (
    HideAction,
    RestAction,
    SearchFoodAction,
    WanderAction,
)


class NeedsAgent(CellAgent, BehaviorMixin):
    def __init__(self, model, cell=None):
        super().__init__(model)
        if cell is not None:
            self.cell = cell

        self.hunger = self.random.uniform(0.20, 0.55)
        self.fear = self.random.uniform(0.08, 0.35)
        self.energy = self.random.uniform(0.45, 0.90)

        self.hunger_weight = model.hunger_weight
        self.fear_weight = model.fear_weight
        self.energy_weight = model.low_energy_weight

        self.current_regime = "neutral"
        self.recent_danger = 0.0
        self.is_dead = False
        self.steps_alive = 0

        self.init_behavior_state(
            default_action="rest",
            history_length=model.history_length,
            action_names=["hide", "rest", "search_food", "wander"],
        )

    def coord(self):
        return self.cell.coordinate

    def torus_distance(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dx = min(dx, self.model.width - dx)
        dy = min(dy, self.model.height - dy)
        return dx + dy

    def get_neighbor_cells(self):
        return list(self.cell.neighborhood)

    def get_neighbor_agents(self):
        neighbors = []
        for c in self.get_neighbor_cells():
            neighbors.extend(list(c.agents))
        return neighbors

    def get_neighbor_fear(self):
        neighbors = self.get_neighbor_agents()
        if not neighbors:
            return 0.0
        return sum(n.fear for n in neighbors) / len(neighbors)

    def nearest_target(self, targets):
        if not targets:
            return None
        return min(targets, key=lambda t: self.torus_distance(self.coord(), t))

    def is_on_food(self):
        pos = self.coord()
        return pos in self.model.food_supply and self.model.food_supply[pos] > 0

    def is_in_danger(self):
        return self.coord() in self.model.danger_positions

    def consume_food(self):
        pos = self.coord()
        if pos in self.model.food_supply and self.model.food_supply[pos] > 0:
            self.model.food_supply[pos] -= 1
            return True
        return False

    def move_randomly(self):
        neighbors = self.get_neighbor_cells()
        if neighbors:
            self.cell = self.random.choice(neighbors)

    def move_towards_food(self):
        active_food = [p for p, amount in self.model.food_supply.items() if amount > 0]
        if not active_food:
            self.move_randomly()
            return

        target = self.nearest_target(active_food)
        neighbors = self.get_neighbor_cells()
        if not neighbors:
            return

        best_cell = min(neighbors, key=lambda c: self.torus_distance(c.coordinate, target))
        self.cell = best_cell

    def move_to_safer_cell(self):
        neighbors = self.get_neighbor_cells()
        if not neighbors:
            return

        def danger_score(cell):
            coord = cell.coordinate
            score = 0.0
            if coord in self.model.danger_positions:
                score += 10.0
            nearest_danger = self.nearest_target(self.model.danger_positions)
            if nearest_danger is not None:
                score += 1 / (1 + self.torus_distance(coord, nearest_danger))
            return score

        safest = min(neighbors, key=danger_score)
        self.cell = safest

    def get_action_cost(self, action_name):
        if action_name == "search_food":
            return 0.04 * (1.0 - self.energy)
        if action_name == "hide":
            return 0.02
        return 0.0

    def update_regime(self):
        if self.fear > 0.60:
            self.current_regime = "fear"
        elif self.hunger > 0.70:
            self.current_regime = "survival"
        elif self.energy < 0.30:
            self.current_regime = "recovery"
        else:
            self.current_regime = "neutral"

    def adapt_weights(self):
        if self.current_regime == "fear":
            self.fear_weight += 0.20
        elif self.current_regime == "survival":
            self.hunger_weight += 0.20
        elif self.current_regime == "recovery":
            self.energy_weight += 0.20

        self.normalize_weights(
            ["hunger_weight", "fear_weight", "energy_weight"],
            floor=0.05,
        )

        self.hunger_weight = min(self.hunger_weight, 0.70)
        self.fear_weight = min(self.fear_weight, 0.70)
        self.energy_weight = min(self.energy_weight, 0.70)

        self.normalize_weights(
            ["hunger_weight", "fear_weight", "energy_weight"],
            floor=0.05,
        )

    def update_internal(self):
        self.steps_alive += 1

        self.hunger = min(1.0, self.hunger + self.model.hunger_drift)
        self.energy = max(0.0, self.energy - self.model.energy_decay)

        if self.is_in_danger():
            self.fear = min(1.0, self.fear + self.model.danger_impact)
            self.recent_danger = min(1.0, self.recent_danger + 0.4)
            self.model.total_danger_exposure += 1
        else:
            self.fear = max(0.03, self.fear * self.model.fear_persistence)
            self.recent_danger *= 0.90

        neighbor_fear = self.get_neighbor_fear()
        if neighbor_fear > self.model.panic_threshold:
            self.fear = min(1.0, self.fear + self.model.contagion_strength)

        if self.energy <= 0.0 or self.hunger >= 1.0:
            self.is_dead = True

    def step(self):
        if self.is_dead:
            return

        self.update_internal()
        self.update_regime()
        self.adapt_weights()

        if self.fear > self.model.hide_interrupt_threshold:
            scores = self.model.policy.evaluate(self)
            probs = {k: 0.0 for k in scores}
            probs["hide"] = 1.0
            action = next(a for a in self.model.actions if a.name == "hide")

            self.last_action = action.name
            self.last_scores = scores
            self.last_probabilities = probs

            self.act(action)
            self.update_behavior_history(action.name)
        else:
            self.behavioral_step()


class NeedsModel(Model):
    def __init__(
        self,
        width=20,
        height=20,
        n_agents=60,
        n_food=30,
        n_danger=18,
        hunger_weight=1.0,
        fear_weight=1.1,
        low_energy_weight=0.9,
        wander_bias=0.08,
        food_gain=0.55,
        hunger_drift=0.03,
        energy_decay=0.025,
        danger_impact=0.14,
        fear_persistence=0.985,
        panic_threshold=0.45,
        contagion_strength=0.06,
        memory_penalty_weight=0.18,
        hide_interrupt_threshold=0.85,
        temperature=0.12,
        score_scale=6.0,
        history_length=8,
        rng=None,
    ):
        super().__init__(rng=rng)

        self.width = width
        self.height = height

        self.hunger_weight = hunger_weight
        self.fear_weight = fear_weight
        self.low_energy_weight = low_energy_weight
        self.wander_bias = wander_bias

        self.food_gain = food_gain
        self.hunger_drift = hunger_drift
        self.energy_decay = energy_decay
        self.danger_impact = danger_impact
        self.fear_persistence = fear_persistence
        self.panic_threshold = panic_threshold
        self.contagion_strength = contagion_strength

        self.memory_penalty_weight = memory_penalty_weight
        self.hide_interrupt_threshold = hide_interrupt_threshold
        self.history_length = history_length

        self.grid = OrthogonalMooreGrid(
            (width, height),
            torus=True,
            capacity=10,
            random=self.random,
        )

        self.actions = [
            HideAction(),
            RestAction(),
            SearchFoodAction(),
            WanderAction(),
        ]
        self.policy = SoftmaxPolicy(
            self.actions,
            temperature=temperature,
            score_scale=score_scale,
        )

        self.food_supply = {}
        self.danger_positions = set()
        self._initialize_environment(n_food=n_food, n_danger=n_danger)

        self.total_food_consumed = 0
        self.total_danger_exposure = 0

        all_cells = [self.grid[(x, y)] for x in range(width) for y in range(height)]
        chosen_cells = self.random.choices(all_cells, k=n_agents)

        for cell in chosen_cells:
            agent = NeedsAgent(self, cell=cell)
            self.agents.add(agent)

        self.datacollector = DataCollector(
            model_reporters={
                "Hiding": lambda m: sum(a.last_action == "hide" and not a.is_dead for a in m.agents),
                "Resting": lambda m: sum(a.last_action == "rest" and not a.is_dead for a in m.agents),
                "Searching Food": lambda m: sum(a.last_action == "search_food" and not a.is_dead for a in m.agents),
                "Wandering": lambda m: sum(a.last_action == "wander" and not a.is_dead for a in m.agents),

                "Avg Hunger": lambda m: _safe_avg(m, lambda a: a.hunger),
                "Avg Fear": lambda m: _safe_avg(m, lambda a: a.fear),
                "Avg Energy": lambda m: _safe_avg(m, lambda a: a.energy),

                "Avg Hide Score": lambda m: _safe_avg(m, lambda a: a.last_scores["hide"]),
                "Avg Rest Score": lambda m: _safe_avg(m, lambda a: a.last_scores["rest"]),
                "Avg Search Score": lambda m: _safe_avg(m, lambda a: a.last_scores["search_food"]),
                "Avg Wander Score": lambda m: _safe_avg(m, lambda a: a.last_scores["wander"]),

                "Avg Hide Prob": lambda m: _safe_avg(m, lambda a: a.last_probabilities["hide"]),
                "Avg Rest Prob": lambda m: _safe_avg(m, lambda a: a.last_probabilities["rest"]),
                "Avg Search Prob": lambda m: _safe_avg(m, lambda a: a.last_probabilities["search_food"]),
                "Avg Wander Prob": lambda m: _safe_avg(m, lambda a: a.last_probabilities["wander"]),

                "Avg Hunger Weight": lambda m: _safe_avg(m, lambda a: a.hunger_weight),
                "Avg Fear Weight": lambda m: _safe_avg(m, lambda a: a.fear_weight),
                "Avg Energy Weight": lambda m: _safe_avg(m, lambda a: a.energy_weight),

                "Survival Rate": lambda m: _survival_rate(m),
                "Total Food Consumed": lambda m: m.total_food_consumed,
                "Total Danger Exposure": lambda m: m.total_danger_exposure,

                "Survival Regime": lambda m: sum(a.current_regime == "survival" and not a.is_dead for a in m.agents),
                "Fear Regime": lambda m: sum(a.current_regime == "fear" and not a.is_dead for a in m.agents),
                "Recovery Regime": lambda m: sum(a.current_regime == "recovery" and not a.is_dead for a in m.agents),
            }
        )
        self.datacollector.collect(self)

    def _initialize_environment(self, n_food, n_danger):
        coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        self.random.shuffle(coords)

        food_coords = coords[:n_food]
        danger_coords = coords[n_food:n_food + n_danger]

        for coord in food_coords:
            self.food_supply[coord] = self.random.randint(2, 5)

        for coord in danger_coords:
            self.danger_positions.add(coord)

    def _regenerate_food(self):
        if not self.food_supply:
            return
        food_coords = list(self.food_supply.keys())
        regen_coord = self.random.choice(food_coords)
        self.food_supply[regen_coord] = min(5, self.food_supply[regen_coord] + 1)

    def step(self):
        self.agents.shuffle_do("step")
        self.policy.decay_temperature()
        self._regenerate_food()
        self.datacollector.collect(self)


def _living_agents(model):
    return [a for a in model.agents if not a.is_dead]


def _safe_avg(model, fn):
    living = _living_agents(model)
    if not living:
        return 0.0
    return sum(fn(a) for a in living) / len(living)


def _survival_rate(model):
    if len(model.agents) == 0:
        return 0.0
    return len(_living_agents(model)) / len(model.agents)