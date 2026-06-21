import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool

# Load API keys
load_dotenv()

# Define our structured output
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Cache the agent so it doesn't rebuild on every button click
@st.cache_resource
def initialize_agent():
    llm = ChatOpenRouter(model="meta-llama/llama-3.3-70b-instruct")
    parser = PydanticOutputParser(pydantic_object=ResearchResponse) 
    
    prompt = ChatPromptTemplate.from_messages([
        (
           "system", 
           "You are a helpful research assistant. Use the Search tool to look up information."
        ),
        ("placeholder","{chat_history}"),
        (
            "human", 
            """Please research this query: {query}
            
            When you are done researching, do NOT use the Search tool again. 
            Respond directly to me with the final answer in the following format:
            
            {format_instructions}
            """
        ),
        ("placeholder", "{agent_scratchpad}")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    tools = [search_tool]
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools, verbose=True), parser

# Initialize the backend
agent_executor, parser = initialize_agent()

# --- STREAMLIT FRONTEND UI ---

st.set_page_config(page_title="AI Research Agent", page_icon="🕵️‍♂️", layout="centered")

st.title("🕵️‍♂️ Autonomous Research Agent")
st.markdown("Ask any question. The AI will search the web, synthesize the data, and generate a structured report.")
st.divider() 

# User input box
query = st.text_input("What would you like to research today?", placeholder="e.g., The history of the first computer...")

# The Search Button
if st.button("Start Research", type="primary"):
    if query:
        # Show a loading spinner while the agent works
        with st.spinner("Agent is searching the web and reading articles..."):
            try:
                # Run the agent backend
                raw_response = agent_executor.invoke({"query": query})
                output_string = raw_response.get("output", "")
                
                # The Safety Net
                if not output_string.strip():
                    st.error("⚠️ The AI got confused and returned an empty response. Please try again.")
                else:
                    # Parse the data
                    structured_response = parser.parse(output_string)
                    
                    # Display the results
                    st.success("Research Complete!")
                    st.header(structured_response.topic)
                    st.subheader("Summary")
                    st.info(structured_response.summary)
                    
                    st.subheader("Sources")
                    for source in structured_response.sources:
                        st.markdown(f"- {source}")
                        
                    st.divider()
                    st.caption(f"**Tools Used:** {', '.join(structured_response.tools_used)}")
                    
            except Exception as e:
                st.error("⚠️ An error occurred while parsing the AI's data.")
                with st.expander("See raw error details"):
                    st.write(e)
                    st.code(output_string)
    else:
        st.warning("Please type a topic into the search box first!")