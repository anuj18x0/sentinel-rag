import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

    async def generate_response(self, question: str, context: str, history: list = None):
        prompt = self._build_prompt(question, context, history)
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    def _build_prompt(self, question: str, context: str, history: list = None):
        history_str = ""
        if history:
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in history])
            history_str = f"\n\nConversation History:\n{history_str}"

        prompt = f"""You are an API reliability assistant.

Use ONLY the provided context.

If the context does not contain enough information to determine a root cause, do not just say "insufficient data". Explain that there is not enough historical or incident data to determine the exact cause, but summarize what the metrics DO show.

If the user is asking a general question about metrics (e.g. "what is the average", "highest latency"), answer directly based on the context without providing probable cause or recommended action.

For incident or root cause questions, provide:
1. probable cause
2. confidence level
3. recommended action

Context:
{context}

{history_str}

User Question: {question}"""
        return prompt

llm_client = LLMClient()
