from .level1 import register_level1_callbacks
from .level2 import register_level2_callbacks, update_level2_logic
from .level3 import register_level3_callbacks, update_level3_logic
from .help import register_help_callbacks

# update_level2_logic / update_level3_logic are re-exported for the test suite.
__all__ = [
    "register_callbacks",
    "register_level1_callbacks",
    "register_level2_callbacks",
    "register_level3_callbacks",
    "register_help_callbacks",
    "update_level2_logic",
    "update_level3_logic",
]

def register_callbacks(app):
    register_level1_callbacks(app)
    register_level2_callbacks(app)
    register_level3_callbacks(app)
    register_help_callbacks(app)
