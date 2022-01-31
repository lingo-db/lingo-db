import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import webbrowser
from profile_data import ProfileData, OpColorMap
from hierarchical_sunburst import hierarchical_sunburst
from opgraph import opgraph
from swimlane import create_swimline_chart
from memory_heatmap import memory_heatmaps
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
                    dbc.NavItem(dbc.NavLink("Overview", active='exact', href="/overview")),
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
                            dbc.Col(html.I(className="fa fa-solid fa-memory"), ),
                            dbc.Col(" Memory"),
                        ],
                        align="center",
                        className="g-0",
                    ), active='exact', href="/memory")),
                    dbc.NavItem(dbc.NavLink(dbc.Row(
                        [
                            dbc.Col(html.I(className="fa fa-solid fa-code"), ),
                            dbc.Col(" MLIR"),
                        ],
                        align="center",
                        className="g-0",
                    ), active='exact', href="/mlir")),

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
################################################  Memory  ##############################################################
########################################################################################################################
memoryHeatmapsData = memory_heatmaps(data)
memory = html.Div(children=dbc.Row(list(map(lambda x: dbc.Col(dbc.Card(
    dbc.CardBody(
        [
            html.H5(x[1], style={'textAlign': 'center'}),
            dcc.Graph(
                id='mem' + str(x[0]),
                figure=x[2],
                style={"height": "35vh"},
                # config={'staticPlot': True}

            )
        ],
    )
), width=4), memoryHeatmapsData))))

########################################################################################################################
#################################################  MLIR  ###############################################################
########################################################################################################################

MLIRPage = html.Div(children=dbc.Row(
    [dbc.Col(createPerfCodeView(data, 3, colorLineMap=createColorLineMap(data, 3, colo_map)), width=8),
     dbc.Col(opgraph(data, colo_map, True), width=4)]))


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/overview':
        return overviewPage
    elif pathname == '/memory':
        return memory
    elif pathname == '/mlir':
        return MLIRPage
    elif pathname == '/other':
        return alternative
    else:
        return overviewPage


def open_browser():
    webbrowser.open_new("http://localhost:{}".format(8050))


if __name__ == '__main__':
    # Timer(1, open_browser).start();
    app.run_server(debug=True)
