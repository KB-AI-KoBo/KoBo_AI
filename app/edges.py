from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph

from AgentState import AgentState
from typing import List, Dict, Literal
import os

# edge function

# agent가 어디서 검색을 원하는지 파악함
def which_retrieved(state: AgentState) -> str:
    # agent response을 통해 검색 경로를 결정
    agent_response = state.get('agent_response', '').lower()
    if "유저 파일" in agent_response: return "user_file"
    elif "데이터베이스" in agent_response: return "db"
    else: return "user_file"  # default
    
def decide_drawing_graph(state:AgentState) -> str:
    print('...그래프 결정 중...')
    agent_response = state.get('agent_response','').lower().strip()

    if "그래프 데이터 준비" in agent_response: 
        print('...결정 완료...')
        return "yes"
    else: return "no"

# generated_answer를 다시 작성할지 질문과 답변을 기반으로 판단
def grade_documents(state: AgentState) -> str:
    
    print("---CHECK ANSWER---")
    
    # 입력 유효성 검사
    if not state or 'input' not in state or 'generated_answer' not in state:
        print("---ERROR: Invalid state or missing required fields---")
        return "no"

    generated_answer = state.get('generated_answer')
    
    # generated_answer 유효성 검사
    if not generated_answer:
        print("---ERROR: No generated_answer found---")
        return "no"

    question = state['input']

    model = state['llm']


    prompt = PromptTemplate.from_template("""
    You are evaluating AI answer to human question.
    answer: {answer}
    Question: {question}
    agent_response: {agent_response}
    agent_response is not answer to the question. Please be careful.
    If the generated text contains false information and toxic words, say no.
    If the generted text contains information to answer human question, say yes.
    Respond with only 'yes' or 'no' to indicate relevance.
    """)

    chain = prompt | model.bind(temperature = 0.3)

    # try 문을 통한 error 처리
    try:
        response = chain.invoke({"question": question, "answer": generated_answer, "agent_response":state.get('agent_response', [])})
        # yes or no의 답변을 통해 판단
        grade = response.content.strip().lower()
        # 다른 답변이 나올 경우 defalt로 no 처리
        if grade not in ["yes", "no"]:
            print(f"---WARNING: Unexpected response '{grade}', defaulting to 'no'---")
            grade = "no"
    except Exception as e:
        print(f"---ERROR: Failed to process - {str(e)}---")
        grade = "no"

    print(f"---DECISION: DOCS {'GOOD' if grade == 'yes' else 'NOT ENOUGH'}---")
    return grade

# 계속되는 iteration을 대비해 Graph 순환을 5번 아래로 제한
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    if state.get('iteration_count', 0) > 5:
        return "end"
    state['iteration_count'] = state.get('iteration_count', 0) + 1
    return "continue"
