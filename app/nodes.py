# import library
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.tools import Tool
from langchain.vectorstores import FAISS
from typing import List, Dict
from langchain.prompts import MessagesPlaceholder
from AgentState import AgentState
from typing import List, Dict, Any
from langchain_experimental.utilities import PythonREPL
import matplotlib.pyplot as plt
import base64
import io


# 1. agent node
def agent(state):

    llm = state['llm']

    # ChatPromptTemplate을 이용, 답변 생성에 필요한 다양한 프롬프트를 입력함과 동시에 agent가 상호작용하며 작동할 수 있게 함
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant helping small business with financial information retrieval and generation. 
         1. Please Answer in Korean. 
         2. Make sure especially yourself generate right answer on the given information. 
         3. You must not invoke fuction. No Invoking.
         4. if you need to retrieve information, put the word 유저 파일 or 데이터베이스. In most cases, when user asks about his input file or his own company or business,you need to retrieve user file.
         5. In other cases, for example yser file isn't, you need to retrieve database.
         6. Analyze user input and write description of the data that you need. 
         7. Make your response easy to use search word for vector db. Just write words or sentences.'''),
        ("human", "{input}"),
        ("ai", "I understand. I'll determine the best course of action."),
        ("human", "Great, what do you think we should do next?"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    # agent가 이용할 도구 정의
    tools = [
        Tool(name="input_retrieve", func=input_retrieve, description="Retrieve information from user-provided files"),
        Tool(name="db_retrieve", func=db_retrieve, description="Retrieve information from the database"),
    ]
    # agent 정의
    agent = OpenAIFunctionsAgent(llm=llm, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.run(state['input'])
    return {"agent_response": response}

# 2. user's input retrieve node
def input_retrieve(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    # 입력이 dict 타입이 아닌 경우 예외 처리
    if not isinstance(state, dict) and not hasattr(state, 'get'):
        raise TypeError(f"Expected state to be a dict or have a 'get' method, but got {type(state)}")  

    # vectorstore 가져오기
    vectorstore = state.get('pdf_db', None)

    # vectorstore가 None인 경우 작업 스킵
    if vectorstore is None:
        print("vectorstore is None, skipping retrieval.")
        return {"retrieved_docs": []}  # 빈 리스트 반환

    # agent_response 가져오기
    agent_response = state.get('agent_response', '').lower()

    # retriever 설정 및 문서 검색
    retriever = vectorstore.as_retriever(k=7)
    docs = retriever.invoke(agent_response)
    print("PDF retrieved and ready")
    
    return {"retrieved_docs": docs}

# 4. DB retrieve node
def db_retrieve(state: AgentState) -> Dict[str, List[Dict[str, Any]]] :
    if not isinstance(state, dict) and not hasattr(state, 'get'):
        raise TypeError(f"Expected state to be a dict or have a 'get' method, but got {type(state)}")
    supporting_db = state.get('supporting_db', FAISS)
    agent_response = state.get('agent_response', '').lower()
    retriever = supporting_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.5}, k=8)
    
    docs = retriever.invoke(agent_response)
    return {"db_docs": docs}

# 5. combiner node
# 검색 결과를 합쳐주는 노드. 이원화된 db에서 정보를 가져와 답변할 수 있도록 함
def combiner(state: AgentState) -> Dict[str, List[Dict[str, Any]]]:
    retrieved_docs = state.get('retrieved_docs', []) or []
    db_docs = state.get('db_docs', []) or []
    combined_result = retrieved_docs + db_docs
    print('combiner complete')
    return {"combined_result": combined_result}

# 6 . rewrite node
# 그래프를 한번 더 순환하게 될 때, 질문을 검색한 텍스트를 이용해 증강해주는 agent node
def rewrite(state: AgentState) -> Dict[str, str]:
    combined_result = state.get('combined_result', [])
    combined_text = " ".join([doc.page_content for doc in combined_result])
    llm = state['llm']
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant that rewrites question from the user for AI agent to make the answer more accurate and perfect.
                      Write the subtle question based on given information.
                      And please write description about data that agent needs to find.'''),
        ("human", "Please rewrite the following information: {context}, question:{input}, AI answer:{answer}"),
    ])
    chain = prompt | llm.bind(temperature = 0.5)
    rewritten_info = chain.invoke({"input": state['input'], "context":combined_text,"answer":state["generated_answer"]})
    return {"input": rewritten_info.content}



# 7. generate node
# 사용자를 위한 답변을 생성하는 agent node
def generate(state: AgentState) -> Dict[str, str]:
    combined_result = state.get('combined_result', [])
    combined_text = ' '.join(doc.page_content for doc in combined_result)

    llm = state['llm']
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant that generates the answer based on given context. 
                      Please write in Korean. Answer logically and avoid writing false information'''),
        ("human", '''Using the following information, generate a comprehensive response on the question.
                    구체적인 수치를 인용하며 서술해라. 지원 사업 데이터에 링크가 있다면 참조해줘.
                    기록번호나 날짜는 제외해서 서술해도 돼.
                    your job is not drawing graph. Only describe it.
                information:{context}, question: {query}, agent_response : {agent_response}'''),
    ])
    
    chain = prompt | llm.bind(temperature = 0.7)
   
    generated_info = chain.invoke({"context": combined_text, "query":state.get('input'), "agent_response":state.get('agent_response', [])}) 
    return {"generated_answer": generated_info.content}

