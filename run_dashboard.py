import sys
import os

# Ensure the root directory is in the path so we can run the app easily
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from dashboard.app import app

if __name__ == '__main__':
    app.run(debug=True, port=8050)
