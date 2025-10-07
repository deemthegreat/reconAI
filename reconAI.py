import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEndpoint

# --- DEFINE A DUCKDUCKGO SEARCH TOOL ---
from crewai_tools import Tool
from duckduckgo_search import DuckDuckGoSearch  # updated import

class DuckDuckGoSearchTool(Tool):
    def run(self, query):
        search = DuckDuckGoSearch()
        results = search.search(query, max_results=5)
        if not results:
            return "No results found."
        return "\n".join([f"{r['title']}: {r['href']}" for r in results])

# Instantiate the search tool
search_tool = DuckDuckGoSearchTool()

# --- SETUP THE LOCAL LLM ---
os.environ["HUGGINGFACE_API_KEY"] = "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"

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512
)

# --- AGENT DEFINITIONS ---
recon_agent = Agent(
    role='Digital Reconnaissance Specialist',
    goal='To search the public internet for any mentions, social media profiles, and online activities related to a given {topic}.',
    backstory=(
        "You are an expert in Open Source Intelligence (OSINT). "
        "Your mission is to find every piece of public data about a person, "
        "focusing on social media, forums, and public websites."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

analyst_agent = Agent(
    role='Data Clustering and Profiling Analyst',
    goal='To analyze the raw data from the reconnaissance agent and group it into logical clusters like "Professional," "Personal," or "Hobbies."',
    backstory=(
        "You are a skilled data analyst. You see patterns where others see chaos. "
        "Your job is to take a messy list of links and data points and organize them "
        "into a clean, structured profile that tells a story about the person's digital life."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

security_agent = Agent(
    role='Digital Security Risk Assessor',
    goal='To assess the clustered digital footprint and assign a security risk score from 1 (very safe) to 10 (high risk). Provide actionable advice to mitigate risks.',
    backstory=(
        "You are a cybersecurity expert with a focus on personal digital protection. "
        "You evaluate how public information could be exploited by bad actors. "
        "Your final report must include a clear risk score, a justification for that score, "
        "and three simple, concrete steps the person can take to improve their digital security."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- TASK DEFINITIONS ---
recon_task = Task(
    description='Conduct a thorough digital reconnaissance on the provided {topic}. Gather all public information, links, social media profiles, and any other relevant data.',
    expected_output='A raw, unorganized list of all data points, links, and text snippets found.',
    agent=recon_agent
)

analysis_task = Task(
    description='Analyze the raw data provided and cluster it into logical groups. For example: "Professional Life," "Social Media Presence," "Forum Activity," "Personal Hobbies." Create a structured report of these clusters.',
    expected_output='A well-structured markdown report with clear headings for each cluster and the corresponding data points listed under them.',
    agent=analyst_agent
)

security_task = Task(
    description='Based on the clustered report, conduct a security risk assessment. Assign an overall digital risk score from 1 to 10. Justify the score and provide a bulleted list of three actionable recommendations for the user to improve their security.',
    expected_output='A final report containing the overall risk score, a paragraph explaining the reasoning, and a list of three security recommendations.',
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
st.markdown("This tool uses a team of AI agents to find and analyze your public digital footprint for **free**, running entirely on your local machine.")

st.sidebar.header("About")
st.sidebar.info("This app uses the CrewAI framework and a local Ollama model (Mistral 7B) to perform its tasks. This ensures your data stays private and the service remains free.")

st.header("Enter a Target for Reconnaissance")
topic_input = st.text_input("Enter a name, username, or email address to investigate:", placeholder="e.g., 'John Doe', 'johndoe123', or 'john.doe@email.com'")

if st.button("🕵️‍♂️ Start Analysis", type="primary"):
    if topic_input:
        with st.spinner("The AI crew is assembling and beginning the investigation... This may take a few minutes."):
            st.info("**Phase 1: Reconnaissance** - The first agent is searching the web...")

            # Kick off the crew's work
            result = digital_footprint_crew.kickoff(inputs={'topic': topic_input})

            st.success("Analysis Complete!")
            st.markdown("---")
            st.header("Final Digital Footprint Report")
            st.markdown(result)
    else:
        st.error("Please enter a target to investigate.")
