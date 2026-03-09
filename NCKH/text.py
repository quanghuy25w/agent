from src.tools.system_prompt import search_ctdt
# Cách viết chuẩn hiện nay
from llama_index.core import VectorStoreIndex

print(
    search_ctdt("Danh sách học phần lựa chọn CNTT K19")
)