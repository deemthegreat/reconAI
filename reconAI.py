import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEndpoint
from duckduckgo_search import ddg_answers  # Cloud-friendly

# --- SETUP THE HUGGING FACE LLM ---
os.environ["HUGGINGFACE_API_KEY"] = "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"

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=512
)

# --- DEFINE CLOUD-FRIENDLY SEARCH TOOL ---
class DuckDuckGoSearchTool:
    def search(self, query):
        results = ddg_answers(query, max_results=5)
        return "\n".join([r["body"] for r in results if "body" in r])

search_tool = DuckDuckGoSearchTool()

# --- AGENTS ---
recon_agent = Agent(
    role='Digital Reconnaissance Specialist',
    goal='To search the public internet for mentions, social media, and online activities of {topic}.',
    backstory=(
        "You are an expert in OSINT. "
        "Your mission is to find public data about a person, "
        "focusing on social media, forums, and public websites."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

analyst_agent = Agent(
    role='Data Clustering and Profiling Analyst',
    goal='Analyze raw data from reconnaissance and cluster it into logical groups like "Professional," "Personal," or "Hobbies."',
    backstory="You are a skilled data analyst. Your job is to take messy data and create a structured profile.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

security_agent = Agent(
    role='Digital Security Risk Assessor',
    goal='Assess clustered data and assign a digital security risk score from 1 to 10, with mitigation steps.',
    backstory=(
        "You are a cybersecurity expert. "
        "Provide a risk score, justification, and 3 actionable recommendations."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- TASKS ---
recon_task = Task(
    description='Conduct digital reconnaissance for {topic}, gathering all public data.',
    expected_output='Raw list of data points, links, and snippets.',
    agent=recon_agent
)

analysis_task = Task(
    description='Cluster raw data into logical groups and create a structured report.',
    expected_output='Structured markdown report with clusters.',
    agent=analyst_agent
)

security_task = Task(
    description='Perform security assessment on clustered data.',
    expected_output='Risk score, explanation, and 3 security recommendations.',
    agent=security_agent
)

# --- CREW ---
digital_footprint_crew = Crew(
    agents=[recon_agent, analyst_agent, security_agent],
    tasks=[recon_task, analysis_task, security_task],
    process=Process.sequential,
    verbose=2
)

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Digital Footprint Agent", layout="wide")
st.title("🤖 AI Digital Footprint & Risk Score Agent")
st.markdown("Analyze a person's digital footprint safely using AI agents.")

st.sidebar.header("About")
st.sidebar.info("Uses CrewAI framework and Hugging Face API for AI models. Runs fully in the cloud.")

topic_input = st.text_input("Enter a name, username, or email:", placeholder="e.g., 'John Doe'")

if st.button("🕵️‍♂️ Start Analysis"):
    if topic_input:
        with st.spinner("Assembling AI crew and performing investigation..."):
            st.info("Phase 1: Reconnaissance...")
            result = digital_footprint_crew.kickoff(inputs={'topic': topic_input})
            st.success("Analysis Complete!")
            st.markdown("---")
            st.header("Final Digital Footprint Report")
            st.markdown(result)
    else:
        st.error("Please enter a target to investigate.")
