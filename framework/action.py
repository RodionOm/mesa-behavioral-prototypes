class Action:
    """Reusable action abstraction.

    Each action defines:
    - score(agent): how attractive the action is
    - execute(agent): how it changes the agent/environment
    """

    name = "base"

    def score(self, agent):
        raise NotImplementedError

    def execute(self, agent):
        raise NotImplementedError