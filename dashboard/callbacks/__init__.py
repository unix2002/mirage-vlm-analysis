from .level2 import register_level2_callbacks, update_level2_logic
from .level3 import register_level3_callbacks, update_level3_logic, ablate_token_logic

def register_callbacks(app):
    register_level2_callbacks(app)
    register_level3_callbacks(app)
