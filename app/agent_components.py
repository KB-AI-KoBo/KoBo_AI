from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import MessagesPlaceholder
from datetime import datetime


def initialize_agent_components(llm):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    # Prompt 템플릿을 한 번만 초기화
    base_prompt = ChatPromptTemplate.from_messages([
        ("system", f'''You are an AI assistant helping small business with financial information retrieval and generation. 
         1. Please Answer in Korean. 
         2. Make sure especially yourself write right answer on the given information. 
         3. There are three methods for searching information: searching the user file, querying the database, or using a search API.
            Use the user file when the user asks about their company or their file. In this case, print the string '유저 파일'.
            Use the database when the user inquires about SME (Small and Medium Enterprises) support programs. In this case, print the string '데이터베이스'.
            Finally, use the search API for real-time information or data that is unlikely to be found in the other two methods. Print the string '검색엔진'.
         4. Analyze user input and write description of the data that you need. 
         5. Alert: For efficient searching, add [[ at the beginning and ]] at the end of the content to be searched. And when you use search engine, 검색엔진에 적합한 하나의 검색어만 작성해줘. 하나는 무조건 작성해야 돼.
         6. 당신의 답변을 짧고 간결하게 작성해 retriever 사용에 적합하게 하고, 한국어를 사용하세요.
         7. Current time : {current_time}
         지금까지의 채팅 내역입니다. 사용자가 이전 질문에 대해 추가적인 정보를 원할 때는 참고하세요. Chat History: {{history}}'''),
        ("human", "{input}"),
        ("ai", "I understand. I'll determine the best course of action."),
        ("human", "Great, what do you think we should do next?"),
    ])
    
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant that rewrites question from the user for AI agent to make the answer more accurate and perfect.
                      Write the subtle question based on given information.
                      And please write description about data that agent needs to find.
                      '''),
        ("human", "Please rewrite the following information: {context}, question:{input}, AI answer:{answer}")
    ])
    
    generate_prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are an AI assistant that generates the answer based on given context. 
                      Please write in Korean. Answer logically and avoid writing false information'''),
        ("human", '''Using the following information, generate a comprehensive response on the question.
                    When you refer to news, please indicate source link.
                    구체적인 수치를 인용하며 서술해라. 지원 사업 데이터에 링크가 있다면 정확히 참조해줘.
                    기록번호나 날짜는 제외해서 서술해도 돼.
                information:{context}, question: {query}, agent_response : {agent_response}''')
    ])

    grade_prompt = PromptTemplate.from_template("""
    You are evaluating AI answer to human question.
    answer: {answer}
    Question: {question}
    agent_response: {agent_response}
    agent_response is not answer to the question. Please be careful.
    If the generated text contains false information and toxic words, say no.
    If the generted text contains information to answer human question, say yes.
    Respond with only 'yes' or 'no' to indicate relevance.
    """)

    
    # LLM 바인딩을 미리 한 번만 실행
    base_chain = base_prompt | llm.bind(temperature=0.4)
    rewrite_chain = rewrite_prompt | llm.bind(temperature=0.3)
    generate_chain = generate_prompt | llm.bind(temperature=0.6)
    grade_chain = grade_prompt | llm.bind(temperature = 0.3)
    return {
        "base_chain": base_chain,
        "rewrite_chain": rewrite_chain,
        "generate_chain": generate_chain,
        "grade_chain": grade_chain
    }