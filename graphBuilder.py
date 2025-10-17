import json
import networkx as nx


def build_ds_code_graph_from_single_json_file(json_file_path):
    """
    Xây dựng DS-Code Graph từ MỘT tệp JSON duy nhất chứa danh sách các chunk.
    """
    graph = nx.DiGraph()

    print(f"Đang đọc dữ liệu từ tệp: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        all_json_data = json.load(f)  # Tải toàn bộ danh sách

    if not isinstance(all_json_data, list):
        raise TypeError("Tệp JSON phải chứa một danh sách (list) các đối tượng chunk.")

    # --- Lượt 1: Thêm tất cả các nút vào đồ thị ---
    print("Lượt 1: Đang thêm các nút (code chunks)...")
    for data in all_json_data:
        node_id = data['id']
        graph.add_node(
            node_id,
            content=data['content'],
            metadata=data['metadata']
        )
    print(f"Hoàn thành Lượt 1. Đã thêm {graph.number_of_nodes()} nút.")

    # --- Lượt 2: Thêm các cạnh CALLS ---
    print("Lượt 2: Đang thêm các cạnh (quan hệ CALLS)...")
    for data in all_json_data:
        caller_id = data['id']
        if 'calls' in data['metadata'] and data['metadata']['calls']:
            for callee_id in data['metadata']['calls']:
                if graph.has_node(caller_id) and graph.has_node(callee_id):
                    graph.add_edge(caller_id, callee_id, relation="CALLS")
                else:
                    print(f"Cảnh báo: Không tìm thấy nút '{callee_id}' được gọi bởi '{caller_id}'.")

    print(f"Hoàn thành Lượt 2. Đồ thị hiện có {graph.number_of_edges()} cạnh.")
    return graph


# --- Sử dụng ---
JSON_FILE = './chunks_output.json'  # <--- Cung cấp đường dẫn đến TỆP
ds_code_graph = build_ds_code_graph_from_single_json_file(JSON_FILE)