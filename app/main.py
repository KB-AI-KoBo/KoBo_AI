from workflow import run_workflow, extract_final_response
from dotenv import load_dotenv
import os
from build_vector_db import pdf_to_vector_db, public_to_vector_db
from langchain.chat_models import ChatOpenAI
from ExtractLink import ExtractLink
import requests
# recursion_limit
if __name__ == "__main__":
    load_dotenv()
    # API 키 가져오기
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # 에러 처리
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    supporting_db = public_to_vector_db()
    llm = ChatOpenAI(temperature=0.5, model='gpt-4o',openai_api_key= openai_api_key)
    
    while True:
        
        query = input("질문을 입력하세요 (종료하려면 'exit' 입력): ")
        pdf_path = os.path.join("/Users/seun/Desktop/AI-Challenge/KoBo/KoBo/test_data/test_input.pdf")
        pdf_db = pdf_to_vector_db(pdf_path)
        # exit 입력시 반복문 탈출
        if query.lower() == 'exit':
            break
        
        result = run_workflow(query, pdf_path, openai_api_key, pdf_db, supporting_db, llm)
        # LangGraph의 마지막 답변을 추
        response = extract_final_response(result)
        print("답변:", response)
        extracted_data = ExtractLink(response, llm)
        # 백엔드 API 엔드포인트 URL
        backend_url = "http://backend-url.com/api/save-link"

        # 추출된 데이터를 JSON 형식으로 백엔드에 전송
        response = requests.post(backend_url, json=extracted_data)