from dash import Input, Output, State

def register_help_callbacks(app):
    @app.callback(
        Output("help-modal-step1", "is_open"),
        Input("help-btn-step1", "n_clicks"),
        State("help-modal-step1", "is_open"),
        prevent_initial_call=True
    )
    def toggle_modal1(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("help-modal-step2", "is_open"),
        Input("help-btn-step2", "n_clicks"),
        State("help-modal-step2", "is_open"),
        prevent_initial_call=True
    )
    def toggle_modal2(n1, is_open):
        if n1:
            return not is_open
        return is_open

    @app.callback(
        Output("help-modal-step3", "is_open"),
        Input("help-btn-step3", "n_clicks"),
        State("help-modal-step3", "is_open"),
        prevent_initial_call=True
    )
    def toggle_modal3(n1, is_open):
        if n1:
            return not is_open
        return is_open
