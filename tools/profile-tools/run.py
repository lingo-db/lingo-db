import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import webbrowser
from profile_data import ProfileData, OpColorMap
from hierarchical_sunburst import hierarchical_sunburst
from opgraph import opgraph
from swimlane import create_swimline_chart
from codeview import createPerfCodeView, createColorLineMap

data = ProfileData()
relevant_ops = data.getOperations(cols=["id"], level=0, nested_inside="func")
colo_map = OpColorMap(relevant_ops)
from threading import Timer

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.I(className="fa fa-solid fa-dna"), ),
                        dbc.Col(dbc.NavbarBrand("Profile Results", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink(dbc.Row(
                        [
                            dbc.Col(html.I(className="fa fa-solid fa-clock"), ),
                            dbc.Col(" Overview"),
                        ],
                        align="center",
                        className="g-0",
                    ), active='exact', href="/overview")),
                    dbc.NavItem(dbc.NavLink(dbc.Row(
                        [
                            dbc.Col(html.I(className="fa fa-solid fa-code"), ),
                            dbc.Col(" SubOp"),
                        ],
                        align="center",
                        className="g-0",
                    ), active='exact', href="/mlir-subop")),
                    dbc.NavItem(dbc.NavLink(dbc.Row(
                        [
                            dbc.Col(html.I(className="fa fa-solid fa-code"), ),
                            dbc.Col(" Std"),
                        ],
                        align="center",
                        className="g-0",
                    ), active='exact', href="/mlir-std")),
                    dbc.NavItem(dbc.NavLink(dbc.Row(
                        [
                            dbc.Col(html.I(className="fa fa-solid fa-code"), ),
                            dbc.Col(" LLVM"),
                        ],
                        align="center",
                        className="g-0",
                    ), active='exact', href="/mlir-llvm")),
                ]
            ),

            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(

                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)
count_graph = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Samples", style={'textAlign': 'center'}),
            dcc.Graph(
                id='samples-graph',
                figure=hierarchical_sunburst(data, colo_map),
                style={"height": "40vh"}

            )
        ]
    )
)
cache_misses_graph = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Cache misses", style={'textAlign': 'center'}),
            dcc.Graph(
                id='cache-misses-graph',
                figure=hierarchical_sunburst(data, colo_map, "l3_miss"),
                style={"height": "40vh"}

            )
        ],
    )
)
plan = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Plan", style={'textAlign': 'center'}),
            opgraph(data, colo_map),
        ],
        style={"height": "28vh"}

    )
)

alternative = html.H1(children='Alternative'),

app.layout = html.Div(children=[
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

########################################################################################################################
###############################################  Overview  #############################################################
########################################################################################################################

sl_graph = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Swimlane", style={'textAlign': 'center'}),
            dcc.Graph(
                id='sl-graph',
                figure=create_swimline_chart(data, colo_map, ),
                style={"height": "28vh"},
                # config={'staticPlot': True}

            )
        ],
    )
)
sl_graph_norm = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Swimlane", style={'textAlign': 'center'}),
            dcc.Graph(
                id='sl-graph',
                figure=create_swimline_chart(data, colo_map, normalized="fraction"),
                style={"height": "28vh"},
                # config={'staticPlot': True}

            )
        ],
    )
)
overviewPageSamples = html.Div(children=[dbc.Row([
    dbc.Col(count_graph, width=4),
    dbc.Col(dbc.Row([
        dbc.Col(plan, width=12),
        dbc.Col(sl_graph, width=12),
        dbc.Col(sl_graph_norm, width=12),
    ]), width=8),
]),
])

sl_cache_graph = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Swimlane", style={'textAlign': 'center'}),
            dcc.Graph(
                id='sl_cache_graph',
                figure=create_swimline_chart(data, colo_map, "l3_miss"),
                style={"height": "28vh"},
                # config={'staticPlot': True}

            )
        ],
    )
)
sl_cache_graph_norm = dbc.Card(
    dbc.CardBody(
        [
            html.H5("Swimlane", style={'textAlign': 'center'}),
            dcc.Graph(
                id='sl_cache_graph',
                figure=create_swimline_chart(data, colo_map, "l3_miss", normalized="fraction"),
                style={"height": "28vh"},
                # config={'staticPlot': True}

            )
        ],
    )
)
overviewPageCacheMisses = html.Div(children=[dbc.Row([
    dbc.Col(cache_misses_graph, width=4),
    dbc.Col(dbc.Row([
        dbc.Col(plan, width=12),
        dbc.Col(sl_cache_graph, width=12),
        dbc.Col(sl_cache_graph_norm, width=12),
    ]), width=8),
]),
])
overviewPage = tabs = dbc.Tabs(
    [
        dbc.Tab(overviewPageSamples, label="Samples"),
        dbc.Tab(overviewPageCacheMisses, label="Cache misses"),
    ]
)

########################################################################################################################
#################################################  MLIR  ###############################################################
########################################################################################################################

MLIRLLVMPage = html.Div(children=dbc.Row(
    [dbc.Col(createPerfCodeView(data, 4, colorLineMap=createColorLineMap(data, 4, colo_map)), width=8),
     dbc.Col(opgraph(data, colo_map, True), width=4)]))
MLIRStdPage = html.Div(children=dbc.Row(
    [dbc.Col(createPerfCodeView(data, 3, colorLineMap=createColorLineMap(data, 3, colo_map)), width=8),
     dbc.Col(opgraph(data, colo_map, True), width=4)]))
MLIRSubOpPage = html.Div(children=dbc.Row(
    [dbc.Col(createPerfCodeView(data, 1, colorLineMap=createColorLineMap(data, 1, colo_map)), width=8),
     dbc.Col(opgraph(data, colo_map, True), width=4)]))

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/overview':
        return overviewPage
    elif pathname == '/mlir-subop':
        return MLIRSubOpPage
    elif pathname == '/mlir-std':
        return MLIRStdPage
    elif pathname == '/mlir-llvm':
        return MLIRLLVMPage
    elif pathname == '/other':
        return alternative
    else:
        return overviewPage


def open_browser():
    webbrowser.open_new("http://localhost:{}".format(8050))


if __name__ == '__main__':
    # Timer(1, open_browser).start();
    app.run_server(debug=True)
