import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], assets_folder="../assets"
)
server = app.server
