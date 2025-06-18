from dotenv import load_dotenv
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
tavily = TavilySearchResults()

llm = ChatGroq(model="llama-3.3-70b-versatile")

llm_with_tools=llm.bind_tools(tools)


