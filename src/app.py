"""
File chính để chạy ứng dụng Chatbot AI
Chức năng: 
- Tạo giao diện web với Streamlit
- Xử lý tương tác chat với người dùng
- Kết nối với AI model để trả lời
"""

# === IMPORT CÁC THƯ VIỆN CẦN THIẾT ===
import streamlit as st  # Thư viện tạo giao diện web
from dotenv import load_dotenv  # Đọc file .env chứa API key
from seed_data import seed_milvus # Hàm xử lý dữ liệu
from agent import get_retriever as get_openai_retriever, get_llm_and_agent as get_openai_agent
from local_ollama import get_retriever as get_ollama_retriever, get_llm_and_agent as get_ollama_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# === THIẾT LẬP GIAO DIỆN TRANG WEB ===
def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="AI Assistant",  # Tiêu đề tab trình duyệt
        page_icon="💬",  # Icon tab
        layout="wide"  # Giao diện rộng
    )

import re, html
def linkify(text):
    return re.sub(r'(https?://\S+)', r'[\1](\1)', html.escape(text))


# === KHỞI TẠO ỨNG DỤNG ===
def initialize_app():
    """
    Khởi tạo các cài đặt cần thiết:
    - Đọc file .env chứa API key
    - Cấu hình trang web
    """
    load_dotenv()  # Đọc API key từ file .env
    setup_page()  # Thiết lập giao diện

# === THANH CÔNG CỤ BÊN TRÁI ===
def setup_sidebar():
    """
    Tạo thanh công cụ bên trái với các tùy chọn
    """
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        
        # Phần 1: Chọn Embeddings Model
        st.header("🔤 Embeddings Model")
        embeddings_choice = st.radio(
            "Chọn Embeddings Model:",
            ["text-embedding-3-large", "llama3.2"]
        )
        use_ollama_embeddings = (embeddings_choice == "Ollama")
        
        # Phần 2: Cấu hình Data
        st.header("📚 Nguồn dữ liệu")
        data_source = st.radio(
            "Chọn nguồn dữ liệu:",
            ["File Local", "URL trực tiếp"]
        )
        
        # Xử lý nguồn dữ liệu dựa trên embeddings đã chọn
        if data_source == "File Local":
            handle_local_file(use_ollama_embeddings)
            
        # Thêm phần chọn collection để query
        st.header("🔍 Collection để truy vấn")
        collection_to_query = st.text_input(
            "Nhập tên collection cần truy vấn:",
            "data_test",
            help="Nhập tên collection bạn muốn sử dụng để tìm kiếm thông tin"
        )
        
        # Phần 3: Chọn Model để trả lời
        st.header("🤖 Model AI")
        model_choice = st.radio(
            "Chọn AI Model để trả lời:",
            ["Mistral", "Llama3.2"]
        )
        
        return model_choice, collection_to_query

def handle_local_file(use_ollama_embeddings: bool):
    """
    Xử lý khi người dùng chọn tải file
    """
    collection_name = st.text_input(
        "Tên collection trong Milvus:", 
        "data_test",
        help="Nhập tên collection bạn muốn lưu trong Milvus"
    )
    filename = st.text_input("Tên file JSON:", "stack.json")
    directory = st.text_input("Thư mục chứa file:", "data")
    
    if st.button("Tải dữ liệu từ file"):
        if not collection_name:
            st.error("Vui lòng nhập tên collection!")
            return
            
        with st.spinner("Đang tải dữ liệu..."):
            try:
                seed_milvus(
                    'http://localhost:19530', 
                    collection_name, 
                    filename, 
                    directory, 
                    use_ollama=use_ollama_embeddings
                )
                st.success(f"Đã tải dữ liệu thành công vào collection '{collection_name}'!")
            except Exception as e:
                st.error(f"Lỗi khi tải dữ liệu: {str(e)}")



# === GIAO DIỆN CHAT CHÍNH ===
def setup_chat_interface(model_choice):
    st.title("🏠 HomeGenie – Trợ lý bất động sản")
    
    st.caption("Trơj lý AI được hỗ trợ bởi LangChain và ...")
    
    msgs = StreamlitChatMessageHistory(key="lc_msgs")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
        ]
        msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")

    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])

    return msgs

# === XỬ LÝ TIN NHẮN NGƯỜI DÙNG ===
def handle_user_input(msgs, agent_executor):
    """
    Xử lý khi người dùng gửi tin nhắn:
    1. Hiển thị tin nhắn người dùng
    2. Gọi AI xử lý và trả lời
    3. Lưu vào lịch sử chat
    """
    if prompt := st.chat_input("Hãy cho tôi biết nhu cầu tìm kiếm của bạn!"):
        # Lưu và hiển thị tin nhắn người dùng
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        # Xử lý và hiển thị câu trả lời
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            
            # Lấy lịch sử chat
            chat_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages[:-1]
            ]

            # Gọi AI xử lý
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": chat_history
                },
                {"callbacks": [st_callback]}
            )

            # Lưu và hiển thị câu trả lời
            output = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": output})
            msgs.add_ai_message(output)
            st.write(output)
            st.markdown(linkify(output))
# === HÀM CHÍNH ===
def main():
    """
    Hàm chính điều khiển luồng chương trình
    """
    initialize_app()
    model_choice, collection_to_query = setup_sidebar()
    msgs = setup_chat_interface(model_choice)
    
    # Khởi tạo AI dựa trên lựa chọn model để trả lời
    if model_choice == "Qwen":
        retriever = get_openai_retriever(collection_to_query)
        agent_executor = get_openai_agent(retriever, "gpt4")
    else:
        retriever = get_ollama_retriever(collection_to_query)
        agent_executor = get_ollama_agent(retriever)
    
    handle_user_input(msgs, agent_executor)

# Chạy ứng dụng
if __name__ == "__main__":
    main() 