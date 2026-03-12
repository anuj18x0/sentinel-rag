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

        prompt = f"""You are SentinelAI DevOps Assistant, an expert AI embedded in an API monitoring platform.
You analyze real-time monitoring data and retrieved insights to answer questions about system health.

Context from monitoring database and vector store:
{context}

{history_str}

User Question: {question}

Instructions:
1. Be concise, technical, and actionable.
2. Use markdown (bold, bullet points) for readability.
3. If data shows issues, suggest root causes and remediation steps.
4. Base your answer ONLY on the context provided.

Response:"""
        return prompt

llm_client = LLMClient()
