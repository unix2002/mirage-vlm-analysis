from dash.dependencies import Input, Output, State
import dash
from ..data_loader import LOADER
from ..components.level1_landscape import create_level1_landscape

def register_level1_callbacks(app):
    @app.callback(
        Output('level1-scatter', 'figure'),
        [Input('umap-neighbors-slider', 'value'),
         Input('umap-dist-slider', 'value'),
         Input('umap-color-dropdown', 'value'),
         Input('umap-flippers-toggle', 'value'),
         Input('level1-scatter', 'relayoutData')],
        [State('level1-scatter', 'figure')]
    )
    def update_umap(n_neighbors, min_dist, color_metric, highlight_flippers, relayout_data, current_fig):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Check if the trigger was a zoom/pan event
        is_zoom_event = trigger_id == 'level1-scatter' and relayout_data is not None
        
        # If it's a zoom event and we already have a figure, just update the traces
        if is_zoom_event and current_fig:
            # Check for autorange (double click to reset)
            if 'xaxis.autorange' in relayout_data:
                # Reset to macro view
                return create_level1_landscape(LOADER.get_data(), color_metric=color_metric, zoom_level=1.0, highlight_flippers=highlight_flippers)
            
            # Check if it's an actual zoom (we have specific axis ranges)
            if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                x_min = relayout_data['xaxis.range[0]']
                x_max = relayout_data['xaxis.range[1]']
                
                # Approximate zoom level (smaller range means higher zoom)
                # Base UMAP spans roughly [-20, 20], so range of 40 is zoom 1.0
                x_range = x_max - x_min
                zoom_level = 40.0 / max(0.1, x_range)
                
                # Pass the viewport boundaries to cull out-of-bounds mazes for performance
                viewport = {
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': relayout_data.get('yaxis.range[0]', -100),
                    'y_max': relayout_data.get('yaxis.range[1]', 100)
                }
                
                # Regenerate figure with new zoom state, preserving viewport
                new_fig = create_level1_landscape(
                    LOADER.get_data(), 
                    color_metric=color_metric, 
                    zoom_level=zoom_level,
                    viewport=viewport,
                    highlight_flippers=highlight_flippers
                )
                
                # Restore the exact layout ranges so the zoom doesn't reset or jump
                new_fig.update_layout(
                    xaxis=dict(range=[x_min, x_max]),
                    yaxis=dict(range=[viewport['y_min'], viewport['y_max']])
                )
                
                return new_fig
            
            # For pan events without range changes (rare but possible), return current
            return current_fig

        # Otherwise (parameter sliders changed), recompute everything from scratch
        updated_data = LOADER.recompute_umap(n_neighbors, min_dist, use_pca=True)
        new_fig = create_level1_landscape(updated_data, color_metric=color_metric, zoom_level=1.0, highlight_flippers=highlight_flippers)
        
        return new_fig

