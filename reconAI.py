import os
import streamlit as st
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEndpoint

# --- SETUP THE LOCAL LLM ---
os.environ["HUGGINGFACE_API_KEY"] = "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"

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512
)

# --- DEFINE A DUCKDUCKGO SEARCH TOOL ---
class DuckDuckGoSearchTool(BaseModel):
    name: str = "DuckDuckGoSearch"

    def run(self, query: str) -> str:
        from duckduckgo_search import DDGS
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, region='wt-wt', safesearch='Off', timelimit='y', max_results=5):
                    results.append(r['body'])
        except Exception as e:
            results.append(f"Error during search: {e}")
        return "\n".join(results)

search_tool = DuckDuckGoSearchTool()

# --- AGENT DEFINITIONS ---
recon_agent = Agent(
    role='Digital Reconnaissance Specialist',
    goal='Search the public internet for mentions, social media, and online activities of {topic}.',
    backstory="You are an OSINT expert tasked with gathering public information about a person.",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

analyst_agent = Agent(
    role='Data Clustering and Profiling Analyst',
    goal='Analyze raw data and group into clusters like "Professional," "Personal," "Hobbies."',
    backstory="You are a skilled data analyst. Organize messy data into a clean structured profile.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

security_agent = Agent(
    role='Digital Security Risk Assessor',
    goal='Assess digital footprint and assign a security risk score from 1 (safe) to 10 (high risk). Provide actionable advice.',
    backstory="You are a cybersecurity expert. Provide risk score and actionable recommendations.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- TASK DEFINITIONS ---
recon_task = Task(
    description='Conduct thorough digital reconnaissance on the provided {topic}. Gather public info, links, social profiles, etc.',
    expected_output='Raw unorganized list of all data points, links, and snippets.',
    agent=recon_agent
)

analysis_task = Task(
    description='Analyze raw data and cluster into logical groups with a structured markdown report.',
    expected_output='Structured markdown report with clear headings and listed data points.',
    agent=analyst_agent
)

security_task = Task(
    description='Conduct security risk assessment based on clustered report. Assign score and recommendations.',
    expected_output='Final report with risk score, justification paragraph, and 3 actionable steps.',
    agent=security_agent
)

# --- CREW DEFINITION ---
digital_footprint_crew = Crew(
    agents=[recon_agent, analyst_agent, security_agent],
    tasks=[recon_task, analysis_task, security_task],
    process=Process.sequential,
    verbose=2
)

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="AI Digital Footprint Agent", layout="wide")
st.title("🤖 AI Digital Footprint & Risk Score Agent")
st.markdown("This tool uses a team of AI agents to analyze your public digital footprint, running locally.")

st.sidebar.header("About")
st.sidebar.info("Uses CrewAI + HuggingFace local model (Mistral 7B) to perform tasks while keeping your data private.")

st.header("Enter a Target for Reconnaissance")
topic_input = st.text_input("Enter a name, username, or email address to investigate:", placeholder="e.g., 'John Doe', 'johndoe123', or 'john.doe@email.com'")

if st.button("🕵️‍♂️ Start Analysis", type="primary"):
    if topic_input:
        with st.spinner("AI crew is starting the investigation... This may take a few minutes."):
            st.info("**Phase 1: Reconnaissance** - The first agent is searching the web...")
            try:
                result = digital_footprint_crew.kickoff(inputs={'topic': topic_input})
                st.success("Analysis Complete!")
                st.markdown("---")
                st.header("Final Digital Footprint Report")
                st.markdown(result)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
    else:
        st.error("Please enter a target to investigate.")
