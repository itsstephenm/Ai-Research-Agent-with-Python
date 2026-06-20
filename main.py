print("Script is starting...")
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent 
from langchain_classic.agents import AgentExecutor
from tools import search_tool

load_dotenv()
print("Connecting to OpenRouter...")

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenRouter(model="meta-llama/llama-3.3-70b-instruct")
parser = PydanticOutputParser(pydantic_object=ResearchResponse) 

prompt = ChatPromptTemplate.from_messages(
    [
        (
           "system", 
            """You are a helpful research assistant. 
Use the Search tool to look up information. 

When you are ready to provide the final answer, simply write a markdown JSON block. 
Output ONLY the JSON and absolutely no other text.

{format_instructions}"""
        ),
        ("placeholder","{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool]
agent = create_tool_calling_agent(
    llm=llm, 
    prompt=prompt,
    tools=tools 
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research today? ")
raw_response = agent_executor.invoke({"query": query})

# --- ENHANCED DEBUGGING BLOCK ---
print("\n--- FULL RAW RESPONSE DICTIONARY ---")
print(raw_response)
print("------------------------------------\n")

output_string = raw_response.get("output", "")

# --- THE SAFETY NET (TRY/EXCEPT) ---
if not output_string.strip():
    print("⚠️ ERROR: The model returned an empty response. It got stuck exiting the tool loop.")
    print("Recommendation: Try asking the question again or slightly rephrasing it.")
else:
    try:
        structured_response = parser.parse(output_string)
        print("--- FINAL PARSED DATA ---")
        print(f"Topic: {structured_response.topic}")
        print(f"Summary: {structured_response.summary}")
        print(f"Sources: {', '.join(structured_response.sources)}")
        print(f"Tools Used: {', '.join(structured_response.tools_used)}")
    except Exception as e:
        print(f"⚠️ PARSING ERROR: The AI did not format the JSON correctly.")
        print(f"Error details: {e}")
        print(f"Raw output was: \n{output_string}")