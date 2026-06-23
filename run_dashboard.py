import sys
import os

# Prevent Numba thread-safety crash when UMAP is called concurrently
# by multiple Dash callbacks. The default 'workqueue' layer is not
# threadsafe; 'tbb' is explicitly threadsafe per Numba docs.
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

# Ensure the root directory is in the path so we can run the app easily
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from dashboard.app import app

if __name__ == '__main__':
    app.run(debug=True, port=8056)
