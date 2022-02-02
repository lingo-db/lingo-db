from math import sqrt

import numpy

from profile_data import ProfileData, gen_colors, OpColorMap
import pandas as pd
import dash
from dash import dcc, html

import plotly


data = ProfileData()

num_colors=10
points=[]
for i in range(0,num_colors):
    points.append(i/float(num_colors))
colors=plotly.colors.sample_colorscale(plotly.colors.sequential.Rainbow, points , low=0.0,
                                                        high=1.0,colortype='tuple')
print(colors)
exit(0)






pd.set_option('display.max_columns', None)

app = dash.Dash(__name__)
import pandas as pd
relevant_ops = data.getOperations(cols=["id"], level=0, nested_inside="func")
colo_map = OpColorMap(relevant_ops)
app.layout = html.Div("test")

if __name__ == '__main__':
    app.run_server(debug=True)
