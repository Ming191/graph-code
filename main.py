import config
from graphManager import CodeGraphManager
from agent import CodeRagAgent


def run_pipeline():
    manager = CodeGraphManager(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD)
    if not manager.driver:
        return

    try:
        node_count_result = manager.run_query("MATCH (n) RETURN count(n) as count")
        if node_count_result[0]['count'] == 0:
            print("ℹ️ Đồ thị trống. Bắt đầu xây dựng...")
            manager.build_ds_code_graph_from_single_file(config.JSON_SINGLE_FILE_PATH)
            manager.add_similarity_edges(threshold=config.SIMILARITY_THRESHOLD)
        else:
            print("ℹ️ Đồ thị đã tồn tại trong Neo4j, bỏ qua bước xây dựng.")

        # Bước 2: Đưa ra yêu cầu và tìm kiếm ban đầu
        user_prompt = "Viết hàm `CANProcessor::setRPM(int newRPM)` để cập nhật giá trị RPM và chuẩn bị frame dữ liệu cho việc gửi đi, tương tự như cách `updateDashboard` xử lý speed."

        initial_context = manager.find_initial_support_codes(user_prompt)
        print("\n--- Ngữ cảnh ban đầu ---")
        print(initial_context)

        # Bước 3: Khởi chạy Agent để sinh code
        agent = CodeRagAgent(manager)
        final_code = agent.run(user_prompt, initial_context)

        print("\n" + "=" * 20 + " CODE CUỐI CÙNG " + "=" * 20)
        print(final_code)
        print("=" * 58)

    finally:
        manager.close()


if __name__ == "__main__":
    run_pipeline()