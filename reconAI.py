import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool

# --- ENVIRONMENT AND API SETUP ---
# It's recommended to use Streamlit secrets for API keys.
# You will need a free API key from https://console.groq.com/keys
# os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
# os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

# For local testing, you can set them directly.
os.environ["GROQ_API_KEY"] = "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" # IMPORTANT: Replace with your actual key
os.environ["SERPER_API_KEY"] = "b4fadaeeda46b090334c4e1b9b313d8fcdff85987cb2029fb76e49fad58453b7" # IMPORTANT: Replace with your actual key


# --- LLM CONFIGURATION ---
# Initialize the Groq LLM
try:
    llm = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama3-8b-8192"
    )
except Exception as e:
    st.error(f"Failed to initialize the language model: {e}")
    st.stop()

# --- TOOL DEFINITION ---
search_tool = SerperDevTool()


# --- AGENT DEFINITIONS ---
recon_agent = Agent(
    role='Digital Reconnaissance Specialist',
    goal='Search the public internet for all mentions, social media profiles, and online activities related to {topic}.',
    backstory=(
        "You are an OSINT expert. Your mission is to gather all publicly available "
        "information about a subject, leaving no stone unturned."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

analyst_agent = Agent(
    role='Data Clustering and Profiling Analyst',
    goal='Analyze the raw data gathered and organize it into a structured profile, grouping related information into logical clusters.',
    backstory=(
        "You are a skilled data analyst with a talent for finding patterns in messy data. Your job is to transform raw "
        "intel into a clean, structured, and easy-to-read profile."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

security_agent = Agent(
    role='Digital Security Risk Assessor',
    goal='Assess the subject\'s digital footprint to identify potential security risks, assign a risk score from 1-10, and provide actionable advice.',
    backstory=(
        "You are a cybersecurity expert specializing in digital footprint analysis, providing clear risk scores "
        "and actionable recommendations."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- TASK DEFINITIONS ---
recon_task = Task(
    description='Conduct thorough digital reconnaissance on the provided {topic}. Gather all public information, links, and social profiles.',
    expected_output='A raw, unorganized list of all data points, links, and snippets found.',
    agent=recon_agent
)

analysis_task = Task(
    description='Analyze the raw data from the reconnaissance phase and cluster it into a structured markdown report.',
    expected_output='A well-structured markdown report with clear headings for each cluster of information.',
    agent=analyst_agent
)

security_task = Task(
    description='Conduct a security risk assessment based on the structured report. Assign a risk score and provide actionable recommendations.',
    expected_output='A final report with a numerical risk score (1-10), a justification paragraph, and a list of 3-5 actionable security steps.',
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
st.set_page_config(page_title="AI Digital Footprint Analyzer", layout="wide")

st.title("🤖 AI Digital Footprint & Risk Score Analyzer")
st.markdown("This tool uses a team of AI agents to analyze a public digital footprint. Enter a target below to begin.")

st.sidebar.header("About This Tool")
st.sidebar.info(
    "This application leverages the CrewAI framework and the Llama3 model (via Groq) "
    "to perform a multi-step analysis of a subject's publicly available digital footprint."
)
st.sidebar.warning("You will need free API keys from Serper.dev and Groq for the agents to work. Please add them to the code or your Streamlit secrets.")


st.header("Enter a Target for Reconnaissance")
topic_input = st.text_input(
    "Enter a name, username, or email address to investigate:",
    placeholder="e.g., 'John Doe', 'johndoe123', or 'john.doe@email.com'"
)

if st.button("🕵️‍♂️ Start Analysis", type="primary"):
    if not os.environ.get("SERPER_API_KEY") or os.environ.get("SERPER_API_KEY") == "YOUR_SERPER_API_KEY" or not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "YOUR_GROQ_API_KEY":
        st.error("API key(s) are missing! Please add your Serper and Groq keys to the code or Streamlit secrets to run the analysis.")
    elif topic_input:
        with st.spinner("The AI crew is starting the investigation... This may take a few minutes."):
            st.info("🔄 **Crew Kickoff:** The investigation has started.")
            
            try:
                result = digital_footprint_crew.kickoff(inputs={'topic': topic_input})
                st.success("Analysis Complete!")
                st.markdown("---")
                st.header("Final Digital Footprint Report")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")
    else:
        st.warning("Please enter a target to investigate.")

