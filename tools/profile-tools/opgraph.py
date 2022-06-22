import dash_cytoscape as cyto
from profile_data import createOpNameFromRepr

cyto.load_extra_layouts()

def opgraph(plot_data,colo_map,vertical=False):
    stylesheet = [
        {
            'selector': 'node',
            'style': {

                "content": "data(label)",
                "font-size": "12px",
                "text-valign": "center",
                "text-halign": "center",
                "background-color": "data(color)",
                "color": "#00",
                "z-index": "10",
                'shape': 'round-rectangle',
                'border-color': 'gray',
                'border-width': '1',
                'border-style': 'solid',
                'width': 'label',
                'height': 'label',
                'padding': '4px'

            }
        },
        {
            'selector': '.terminal',
            'style': {
                'width': 90,
                'height': 80,
                'background-fit': 'cover',
                'background-image': 'data(url)'
            }
        },
        {
            'selector': '.nonterminal',
            'style': {
                'shape': 'rectangle'
            }
        }
    ]
    graph = []
    for op in plot_data.getOperations(["id", "repr", "dependencies"], level=0, nested_inside="func"):
        graph.append({'data': {'id': "id_" + str(op[0]), "label": createOpNameFromRepr(op[1]),"color":colo_map.lookup(op[0])}})
        if op[2] is not None:
            for dep in op[2]:
                graph.append({'data': {'target': "id_" + str(op[0]), "source": "id_" + str(dep)}})
    return cyto.Cytoscape(
        id='cytoscape-two-nodes',
        layout={'name': 'dagre',"rankDir": 'LR' if vertical==False else 'TB','nodeDimensionsIncludeLabels':True},
        style={'width': '100%', 'height': '100%'},
        elements=graph,
        stylesheet=stylesheet
    )