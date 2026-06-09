from dash.dependencies import Input, Output
import dash
from ..data_loader import LOADER
from ..components.level1_landscape import create_level1_landscape

def register_level1_callbacks(app):
    @app.callback(
        Output('level1-scatter', 'figure'),
        [Input('umap-neighbors-slider', 'value'),
         Input('umap-dist-slider', 'value'),
         Input('umap-pca-toggle', 'value'),
         Input('umap-color-dropdown', 'value')]
    )
    def update_umap(n_neighbors, min_dist, use_pca, color_metric):
        # 1. Recompute UMAP with new parameters and stabilizers
        updated_data = LOADER.recompute_umap(n_neighbors, min_dist, use_pca=use_pca)
        
        # 2. Regenerate the figure with the chosen color metric
        new_fig = create_level1_landscape(updated_data, color_metric=color_metric)
        
        return new_fig
