from framework.action import Action


class HideAction(Action):
    name = "hide"

    def score(self, agent):
        return max(
            0.01,
            agent.fear_weight * (agent.fear ** 2)
            + 0.30 * agent.recent_danger
            + 0.20 * agent.get_neighbor_fear()
            + agent.get_action_persistence(self.name)
            - agent.get_action_fatigue(self.name)
            - agent.get_action_cost(self.name),
        )

    def execute(self, agent):
        agent.move_to_safer_cell()
        agent.fear = max(0.03, agent.fear - 0.10)
        agent.energy = max(0.0, agent.energy - 0.01)
        agent.hunger = min(1.0, agent.hunger + 0.01)


class RestAction(Action):
    name = "rest"

    def score(self, agent):
        low_energy = (1.0 - agent.energy) ** 2
        return max(
            0.01,
            agent.energy_weight * low_energy
            + 0.08 * max(0.0, 0.4 - agent.fear)
            + agent.get_action_persistence(self.name)
            - agent.get_action_fatigue(self.name)
            - agent.get_action_cost(self.name),
        )

    def execute(self, agent):
        agent.energy = min(1.0, agent.energy + 0.16)
        agent.hunger = min(1.0, agent.hunger + 0.015)
        agent.fear = max(0.03, agent.fear - 0.03)


class SearchFoodAction(Action):
    name = "search_food"

    def score(self, agent):
        return max(
            0.01,
            agent.hunger_weight * (agent.hunger ** 2)
            - 0.25 * agent.fear
            - 0.15 * (1.0 - agent.energy)
            - 0.10 * agent.get_neighbor_fear()
            + agent.get_action_persistence(self.name)
            - agent.get_action_fatigue(self.name)
            - agent.get_action_cost(self.name),
        )

    def execute(self, agent):
        agent.move_towards_food()

        if agent.is_on_food():
            reward_received = agent.consume_food()
            if reward_received:
                agent.hunger = max(0.0, agent.hunger - agent.model.food_gain)
                agent.energy = max(0.0, agent.energy - 0.03)
                agent.model.total_food_consumed += 1
            else:
                agent.energy = max(0.0, agent.energy - 0.05)
        else:
            agent.energy = max(0.0, agent.energy - 0.06)

        if agent.is_in_danger():
            agent.fear = min(1.0, agent.fear + 0.08)
        else:
            agent.fear = min(1.0, agent.fear + 0.01)


class WanderAction(Action):
    name = "wander"

    def score(self, agent):
        return max(
            0.01,
            agent.model.wander_bias
            + 0.04 * (1.0 - agent.hunger)
            + 0.03 * agent.energy
            - 0.05 * agent.get_neighbor_fear()
            + agent.get_action_persistence(self.name)
            - agent.get_action_fatigue(self.name)
            - agent.get_action_cost(self.name),
        )

    def execute(self, agent):
        agent.move_randomly()
        agent.energy = max(0.0, agent.energy - 0.025)
        agent.fear = max(0.03, agent.fear - 0.01)