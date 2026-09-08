import concurrent.futures
from typing import Dict, List, Optional, Tuple

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

try:
    from langchain_mistralai import ChatMistralAI
    HAS_MISTRAL = True
except Exception:
    HAS_MISTRAL = False

try:
    from langchain_ollama import ChatOllama
    HAS_OLLAMA = True
except Exception:
    HAS_OLLAMA = False

def _content(result):
    return result.content if hasattr(result, "content") else str(result)

class FakeLLM:
    def __init__(self, responses):
        self._responses = responses
        self._i = 0
    def invoke(self, prompt):
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r

def build_client(provider: str, config: dict):
    key = config.get("api_key")
    model = config.get("model", "")
    temp = config.get("temperature", 0.1)
    base_url = config.get("base_url", "http://localhost:11434")
    if not model:
        return None
    if provider in ("openai", "gemini", "groq", "mistral") and not key:
        return None
    if provider == "openai" and HAS_OPENAI:
        return ChatOpenAI(model=model, api_key=key, temperature=temp)
    if provider == "gemini" and HAS_GEMINI:
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=temp)
    if provider == "groq" and HAS_GROQ:
        return ChatGroq(model=model, groq_api_key=key, temperature=temp)
    if provider == "mistral" and HAS_MISTRAL:
        return ChatMistralAI(model=model, mistral_api_key=key, temperature=temp)
    if provider == "ollama" and HAS_OLLAMA:
        return ChatOllama(model=model, base_url=base_url, temperature=temp)
    return None

class MultiLLM:
    def __init__(self, providers: List[str], configs: Dict[str, dict], max_retries: int = 1):
        self.providers = providers
        self.configs = configs
        self.max_retries = max_retries
        self.clients = {}
        for p in providers:
            client = build_client(p, configs.get(p, {}))
            if client is None:
                client = FakeLLM([
                    f"Respuesta simulada del proveedor {p}.",
                    "Respuesta generada por el LLM simulado.",
                ])
            self.clients[p] = client

    def call_one(self, provider: str, prompt: str) -> Tuple[str, str]:
        try:
            out = self.clients[provider].invoke(prompt)
            return provider, _content(out)
        except Exception as e:
            return provider, f"Error en {provider}: {e}"

    def call_parallel(self, prompt: str) -> List[Tuple[str, str]]:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = [pool.submit(self.call_one, p, prompt) for p in self.providers]
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    def _quality_score(self, answer: str) -> float:
        if not answer or answer.startswith("Error"):
            return 0.0
        lower = answer.lower()
        if any(p in lower for p in ["no puedo", "no se", "no tengo", "error"]):
            return 0.2
        return min(1.0, len(answer.split()) / 10.0)

    def select_best(self, prompt: str, responses: List[Tuple[str, str]]):
        if not responses:
            return "none", "", 0.0
        if len(responses) == 1:
            return responses[0][0], responses[0][1], self._quality_score(responses[0][1])
        judge = self.clients.get("openai") or self.clients.get("gemini")
        if judge and not isinstance(judge, FakeLLM):
            judge_prompt = (
                "Dada la pregunta y las siguientes respuestas, indica SOLO el numero de la mejor opcion (1, 2, ...). "
                "Si ninguna es adecuada, responde 0.\n\n"
                f"Pregunta: {prompt}\n\n" +
                "\n\n".join([f"Opcion {i+1} ({p}):\n{r}" for i, (p, r) in enumerate(responses)]) +
                "\n\nMejor opcion:"
            )
            try:
                out = _content(judge.invoke(judge_prompt))
                num = int("".join(c for c in out if c.isdigit())[:1])
                if 1 <= num <= len(responses):
                    idx = num - 1
                    return responses[idx][0], responses[idx][1], self._quality_score(responses[idx][1])
            except Exception:
                pass
        best = max(responses, key=lambda x: self._quality_score(x[1]))
        return best[0], best[1], self._quality_score(best[1])

    def generate(self, prompt: str, mode: str = "router", router_provider: Optional[str] = None, max_retries: Optional[int] = None):
        max_retries = max_retries if max_retries is not None else self.max_retries
        if mode == "router":
            provider = router_provider or (self.providers[0] if self.providers else "fake")
            _, answer = self.call_one(provider, prompt)
            return provider, answer
        if mode == "parallel":
            responses = self.call_parallel(prompt)
            for p, ans in responses:
                if not ans.startswith("Error"):
                    return p, ans
            return "parallel", responses[0][1] if responses else ""
        for attempt in range(max_retries + 1):
            responses = self.call_parallel(prompt)
            provider, answer, score = self.select_best(prompt, responses)
            if score >= 0.5 or attempt == max_retries:
                return provider, answer
            prompt = f"Respuesta actual: {answer}\n\nMejora la respuesta para la pregunta original:\n{prompt}"
        return provider, answer

def get_llm(providers: List[str], configs: Dict[str, dict], max_retries: int = 1):
    if not providers:
        providers = ["fake"]
    return MultiLLM(providers, configs, max_retries)
