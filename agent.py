# agent.py

import re
import openai
from duckduckgo_search import DDGS
import config
from graphManager import CodeGraphManager


class CodeRagAgent:
    def __init__(self, graph_manager: CodeGraphManager):
        self.graph_manager = graph_manager
        openai.api_key = config.OPENAI_API_KEY
        self.tools = {
            "web_search": self._web_search,
            "graph_reason": self._graph_reason
        }
        print("\n--- Tác tử CodeRAG đã sẵn sàng ---")

    def _web_search(self, query: str) -> str:
        print(f"🔎 Đang thực hiện Web Search: '{query}'")
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
        return "\n".join(results) if results else "Không tìm thấy kết quả nào."

    def _graph_reason(self, node_id: str) -> str:
        print(f"🕸️ Đang truy vấn đồ thị từ nút: '{node_id}'")
        query = """
            MATCH (n:CodeChunk {id: $node_id})-[r]-(neighbor)
            RETURN type(r) as relation, neighbor.content as content, neighbor.id as id
            LIMIT 5
        """
        results = self.graph_manager.run_query(query, node_id=node_id)
        if not results: return f"Không tìm thấy nút hoặc không có hàng xóm cho '{node_id}'."

        info = [f"// Quan hệ '{res['relation']}' với '{res['id']}'\n{res['content']}" for res in results]
        return "\n\n".join(info)

    def run(self, user_prompt, initial_context):
        context = initial_context
        history = []
        for i in range(3):
            print(f"\n--- Bước suy luận {i + 1}/3 ---")
            prompt = self._build_prompt(user_prompt, context, history)

            response = openai.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            llm_output = response.choices[0].message.content

            thought, action = self._parse_llm_output(llm_output)
            print(f"🤔 Suy nghĩ: {thought}")
            print(f"⚡ Hành động: {action}")
            history.append(f"Thought: {thought}\nAction: {action}")

            if "FINISH" in action.upper():
                print("\n✅ Tác tử đã quyết định hoàn thành.")
                return self.extract_code_from_action(action)

            try:
                tool_name, tool_input = self._parse_action(action)
                if tool_name in self.tools:
                    observation = self.tools[tool_name](tool_input)
                    print(f"🔬 Kết quả quan sát: {observation[:200]}...")
                    context += f"\n\n/* --- Kết quả từ công cụ {tool_name} --- */\n{observation}"
                    history.append(f"Observation: {observation}")
                else:
                    history.append(f"Observation: Lỗi - công cụ '{tool_name}' không tồn tại.")
            except Exception as e:
                history.append(f"Observation: Lỗi khi thực thi hành động - {e}")

        print("\n⚠️ Đã đạt đến giới hạn. Đang cố gắng sinh code cuối cùng...")
        final_prompt = f"Dựa trên toàn bộ ngữ cảnh sau đây, hãy viết code hoàn chỉnh cho yêu cầu: '{user_prompt}'\n\n{context}"
        response = openai.chat.completions.create(model=config.LLM_MODEL,
                                                  messages=[{"role": "user", "content": final_prompt}])
        return self.extract_code_from_action(response.choices[0].message.content)

    def _build_prompt(self, user_prompt, context, history):
        return f"""Bạn là một AI lập trình viên chuyên nghiệp. Nhiệm vụ của bạn là viết code cho yêu cầu sau: '{user_prompt}'.

Bạn có các công cụ sau:
- `web_search(query: str)`: Tìm kiếm thông tin trên internet.
- `graph_reason(node_id: str)`: Khám phá kho code bằng cách truy vấn các đoạn code liên quan đến một `node_id` đã biết.

Dựa vào ngữ cảnh và lịch sử, hãy suy nghĩ và quyết định hành động tiếp theo theo định dạng:

Thought: [Suy nghĩ của bạn.]
Action: [Hành động bạn sẽ thực hiện. VD: `web_search("câu hỏi")`, `graph_reason("id_của_nút")`, hoặc `FINISH` kèm theo khối code.]

### Ngữ cảnh hiện tại:
{context}

### Lịch sử:
{''.join(history)}
"""

    def _parse_llm_output(self, text: str):
        thought = re.search(r"Thought:\s*(.*)", text, re.DOTALL).group(1).strip() if re.search(r"Thought:",
                                                                                               text) else "Không có suy nghĩ."
        action = re.search(r"Action:\s*(.*)", text, re.DOTALL).group(1).strip() if re.search(r"Action:",
                                                                                             text) else "FINISH"
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.match(r'(\w+)\("([^"]*)"\)', action_text)
        if match: return match.groups()
        raise ValueError("Định dạng hành động không hợp lệ.")

    def extract_code_from_action(self, text: str):
        code_match = re.search(r"```(?:\w+)?\n(.*?)\n```", text, re.DOTALL)
        return code_match.group(1).strip() if code_match else text