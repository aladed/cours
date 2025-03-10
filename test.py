import xml.etree.ElementTree as ET


mxfile = ET.Element("mxfile", host="app.diagrams.net")
diagram = ET.SubElement(mxfile, "diagram", id="internet_shop_schema", name="Page-1")
graph_model = ET.SubElement(diagram, "mxGraphModel", dx="1075", dy="569", grid="1", gridSize="10",
                            guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1",
                            pageScale="1", pageWidth="827", pageHeight="1169", math="0", shadow="0")
root = ET.SubElement(graph_model, "root")
ET.SubElement(root, "mxCell", id="0")
ET.SubElement(root, "mxCell", id="1", parent="0")


def create_table(id, name, x, y):
    table = ET.SubElement(root, "mxCell", id=id, value=name, style="swimlane", vertex="1", parent="1")
    geom = ET.SubElement(table, "mxGeometry", x=str(x), y=str(y), width="160", height="120", as_="geometry")
    return table


tables = {
    "user": create_table("user", "User", 20, 20),
    "product": create_table("product", "Product", 220, 20),
    "deal": create_table("deal", "Deal", 120, 180),
    "pvz": create_table("pvz", "PVZ", 20, 360),
    "delivery": create_table("delivery", "Delivery", 220, 360)
}


def create_relation(source, target):
    return ET.SubElement(root, "mxCell", edge="1", source=source, target=target, parent="1")


relations = [
    create_relation("user", "deal"),
    create_relation("product", "deal"),
    create_relation("deal", "pvz"),
    create_relation("deal", "delivery")
]


file_path = "internet_shop_schema.drawio"
tree = ET.ElementTree(mxfile)
tree.write(file_path, encoding="utf-8", xml_declaration=True)

file_path
