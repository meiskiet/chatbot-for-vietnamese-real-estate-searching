import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain.schema import Document
from dotenv import load_dotenv
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def load_data_from_local(filename: str, directory: str) -> tuple:
    """
    Hàm đọc dữ liệu từ file JSON local
    Args:
        filename (str): Tên file JSON cần đọc (ví dụ: 'data.json')
        directory (str): Thư mục chứa file (ví dụ: 'data_v3')
    Returns:
        tuple: Trả về (data, doc_name) trong đó:
            - data: Dữ liệu JSON đã được parse
            - doc_name: Tên tài liệu đã được xử lý (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    """
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r') as file:
        data = json.load(file)
    print(f'Data loaded from {file_path}')
    # Chuyển tên file thành tên tài liệu (bỏ đuôi .json và thay '_' bằng khoảng trắng)
    return data, filename.rsplit('.', 1)[0].replace('_', ' ')

def seed_milvus(URI_link: str, collection_name: str, filename: str, directory: str, use_ollama: bool = False) -> Milvus:
    """
    Hàm tạo và lưu vector embeddings vào Milvus từ dữ liệu local
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection trong Milvus để lưu dữ liệu
        filename (str): Tên file JSON chứa dữ liệu nguồn
        directory (str): Thư mục chứa file dữ liệu
        use_ollama (bool): Sử dụng Ollama embeddings thay vì OpenAI
    """
    # Khởi tạo model embeddings 
    embeddings = OllamaEmbeddings(
        model="llama3.2"          
    )
    
    # Đọc dữ liệu từ file local
    local_data, doc_name = load_data_from_local(filename, directory)
    
    # 2. Convert data into a list of Documents 
    #    mapping your new JSON fields => the Document/metadata fields
    documents = [
        Document(
            page_content=doc.get("page_content", ""),
            metadata={
                "title": doc["metadata"].get("title", ""),
                "date_posted": doc["metadata"].get("date_posted", ""),
                "price_vnd": doc["metadata"].get("price_vnd", 0.0),
                "area": doc["metadata"].get("area", 0.0),
                "price_per_area": doc["metadata"].get("price_per_area", 0.0),
                "bedrooms": doc["metadata"].get("bedrooms", 0.0),
                "toilets": doc["metadata"].get("toilets", 0.0),
                "direction": doc["metadata"].get("direction", ""),
                "district_county": doc["metadata"].get("district_county", ""),
                "province_city": doc["metadata"].get("province_city", ""),
                "url": doc["metadata"].get("url", ""),
                "doc_name": doc_name,
                #"start_index": doc["metadata"].get("start_index", 0)
             }
        )
        for doc in local_data
    ]
    
    print("documents:", documents)

    # Tạo ID duy nhất cho mỗi document
    uuids = [str(uuid4()) for _ in range(len(documents))]
    for doc, uid in zip(documents, uuids):
        doc.metadata["id"] = uid   

    # Khởi tạo và cấu hình Milvus
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
        drop_old=True  # Xóa data đã tồn tại trong collection
    )
    
    # Thêm documents vào Milvus
    vectorstore.add_documents(documents=documents, ids=uuids)
    print('vector: ', vectorstore)
    return vectorstore


def connect_to_milvus(URI_link: str, collection_name: str) -> Milvus:
    """
    Hàm kết nối đến collection có sẵn trong Milvus
    Args:
        URI_link (str): Đường dẫn kết nối đến Milvus
        collection_name (str): Tên collection cần kết nối
    Returns:
        Milvus: Đối tượng Milvus đã được kết nối, sẵn sàng để truy vấn
    Chú ý:
        - Không tạo collection mới hoặc xóa dữ liệu cũ
        - Sử dụng model 'text-embedding-3-large' cho việc tạo embeddings khi truy vấn
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI_link},
        collection_name=collection_name,
    )
    return vectorstore

def main():
    """
    Hàm chính để kiểm thử các chức năng của module
    Thực hiện:
        1. Test seed_milvus với dữ liệu từ file local 'stack.json'
        2. (Đã comment) Test seed_milvus_live với dữ liệu từ trang web stack-ai
    Chú ý:
        - Đảm bảo Milvus server đang chạy tại localhost:19530
        - Các biến môi trường cần thiết (như OPENAI_API_KEY) đã được cấu hình
    """
    # Test seed_milvus với dữ liệu local
    seed_milvus('http://localhost:19530', 'data_thesis', 'stack.json', 'data', use_ollama=False)

# Chạy main() nếu file được thực thi trực tiếp
if __name__ == "__main__":
    main()