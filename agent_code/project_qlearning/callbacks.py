import os
from .features import state_to_features_qlearning
from .model import QTableAgent
from .utils import ACTIONS


def setup(self):
    """
    Setup callback called once before games start.
    Loads the trained model parameters.
    """
    model_dir = os.path.join(os.path.dirname(__file__), 'model_data')
    self.model_filepath = os.path.join(model_dir, 'model.pkl')

    self.model = QTableAgent()

    if os.path.exists(self.model_filepath):
        loaded = self.model.load(self.model_filepath)
        if loaded and self.logger:
            self.logger.info(f"Q-learning model loaded successfully from {self.model_filepath}")
    else:
        if self.logger:
            self.logger.info("No saved model found, starting with default initialized Q-table.")


def act(self, game_state: dict) -> str:
    """
    Called each step to choose an action.
    """
    if game_state is None:
        return 'WAIT'

    state_features = state_to_features_qlearning(game_state)
    action = self.model.choose_action(state_features, game_state=game_state, is_training=self.train)

    if action not in ACTIONS:
        action = 'WAIT'

    return action
