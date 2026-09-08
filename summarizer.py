def summarize(text: str, multi_llm, mode: str = "router", provider: str = None, max_retries: int = 0) -> str:
    prompt = f"Resume el siguiente texto en una frase corta y clara:\n\n{text[:2000]}"
    _, answer = multi_llm.generate(prompt, mode=mode, router_provider=provider, max_retries=max_retries)
    return answer.strip()
