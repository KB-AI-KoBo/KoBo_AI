from langchain.llms.base import LLM
import ollama

# Ollama LLM 클래스 생성
class OllamaLLM(LLM):
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def _call(self, prompt: str, stop=None):
        response = ollama.chat(model=self.model_name, prompt=prompt)
        return response["text"]
    
    @property
    def _identifying_params(self):
        return {"model_name": self.model_name}

# Ollama LLaMA3 모델 사용 예시
llm = OllamaLLM(model_name="llama3")

# 간단한 프롬프트 예시
prompt = "Explain the theory of relativity in simple terms."

# LLaMA3 모델을 사용해 프롬프트에 답변
response = llm(prompt)

# 결과 출력
print(response)

