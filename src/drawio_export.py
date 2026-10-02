"""Export the edited NetworkX graph as an editable draw.io (diagrams.net) file.

The output is plain mxGraph XML (no extra dependency). Open it at
https://app.diagrams.net or in draw.io Desktop and every shape, label and
connector can be moved, edited and restyled. Node positions come from the
shape bounding boxes found in the photo, so the layout matches the whiteboard;
nodes added by hand in the editor (no bbox) are placed on a grid below.
"""

from __future__ import annotations

from xml.sax.saxutils import escape

import networkx as nx

_SHAPE_STYLE = {
    "box": "rounded=0;whiteSpace=wrap;html=1;",
    "circle": "ellipse;whiteSpace=wrap;html=1;aspect=fixed;",
    "diamond": "rhombus;whiteSpace=wrap;html=1;",
}
_MIN_W, _MIN_H = 80, 40
_DEFAULT_W, _DEFAULT_H = 120, 60


def _attr(value: object) -> str:
    return escape(str(value), {'"': "&quot;", "\n": "&#10;"})


def _geometry(graph: nx.Graph) -> dict[int, tuple[float, float, float, float]]:
    geo: dict[int, tuple[float, float, float, float]] = {}
    missing: list[int] = []
    max_y = 0.0
    for n, data in graph.nodes(data=True):
        bbox = data.get("bbox")
        if bbox and len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
            x, y, w, h = (float(v) for v in bbox)
            w, h = max(w, _MIN_W), max(h, _MIN_H)
            if data.get("shape_type") == "circle":
                w = h = max(w, h)
            geo[n] = (x, y, w, h)
            max_y = max(max_y, y + h)
        else:
            missing.append(n)
    for i, n in enumerate(missing):  # manually added nodes
        geo[n] = (40.0 + (i % 5) * 180.0, max_y + 60.0 + (i // 5) * 120.0, _DEFAULT_W, _DEFAULT_H)
    return geo


def graph_to_drawio(graph: nx.Graph, page_name: str = "Whiteboard") -> str:
    """Return draw.io XML text for ``graph``."""
    directed = graph.is_directed()
    geo = _geometry(graph)
    cells = ['<mxCell id="0" />', '<mxCell id="1" parent="0" />']
    for n, data in graph.nodes(data=True):
        x, y, w, h = geo[n]
        style = _SHAPE_STYLE.get(str(data.get("shape_type", "box")), _SHAPE_STYLE["box"])
        label = data.get("label", f"Node_{n}")
        cells.append(
            f'<mxCell id="n{n}" value="{_attr(label)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" as="geometry" />'
            "</mxCell>"
        )
    arrow = "endArrow=classic;" if directed else "endArrow=none;"
    for i, (u, v) in enumerate(graph.edges()):
        cells.append(
            f'<mxCell id="e{i}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;{arrow}" '
            f'edge="1" parent="1" source="n{u}" target="n{v}">'
            '<mxGeometry relative="1" as="geometry" />'
            "</mxCell>"
        )
    body = "".join(cells)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="Whiteboard_to_editable_diagram">'
        f'<diagram id="page1" name="{_attr(page_name)}">'
        '<mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" '
        'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" '
        'pageHeight="850" math="0" shadow="0">'
        f"<root>{body}</root>"
        "</mxGraphModel></diagram></mxfile>\n"
    )


def export_drawio_bytes(graph: nx.Graph | None) -> bytes:
    if graph is None or graph.number_of_nodes() == 0:
        raise ValueError("Nothing to export — process a photo or add nodes first.")
    return graph_to_drawio(graph).encode("utf-8")
