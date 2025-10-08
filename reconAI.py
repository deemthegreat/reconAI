import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun # Use the stable LangChain community tool

# --- ENVIRONMENT AND API SETUP ---
# It's recommended to use Streamlit secrets for API keys.
# For local testing, you can set it directly.
# Example: os.environ["HUGGINGFACE_API_KEY"] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ["HUGGINGFACE_API_KEY"] = "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"

# --- LLM CONFIGURATION ---
# Initialize the HuggingFaceEndpoint for the Mistral model.
try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=512
    )
except Exception as e:
    st.error(f"Failed to initialize the language model: {e}")
    st.stop() # Stop the app if the LLM can't be loaded.

# --- TOOL DEFINITION ---
# Instantiate the DuckDuckGo search tool from langchain_community.
# This provides a stable and reliable search integration for the agent.
search_tool = DuckDuckGoSearchRun()


# --- AGENT DEFINITIONS ---
# Agent 1: The Reconnaissance Specialist
recon_agent = Agent(
    role='Digital Reconnaissance Specialist',
    goal='Search the public internet for all mentions, social media profiles, and online activities related to {topic}.',
    backstory=(
        "You are an OSINT (Open Source Intelligence) expert. Your mission is to gather all publicly available "
        "information about a subject, leaving no stone unturned on the public web."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

# Agent 2: The Data Analyst
analyst_agent = Agent(
    role='Data Clustering and Profiling Analyst',
    goal='Analyze the raw data gathered and organize it into a structured profile. Group related information into logical clusters like "Professional Life," "Personal Details," and "Online Hobbies."',
    backstory=(
        "You are a skilled data analyst with a talent for finding patterns in messy data. Your job is to take the raw "
        "intel and transform it into a clean, structured, and easy-to-read profile."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Agent 3: The Security Assessor
security_agent = Agent(
    role='Digital Security Risk Assessor',
    goal='Assess the subject\'s digital footprint to identify potential security risks. Assign a risk score from 1 (very safe) to 10 (high risk) and provide clear, actionable advice for improving their digital security.',
    backstory=(
        "You are a cybersecurity expert specializing in digital footprint analysis. You provide a clear risk score "
        "and actionable recommendations to help people protect their online presence."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# --- TASK DEFINITIONS ---
# Task 1: Conduct Reconnaissance
recon_task = Task(
    description='Conduct a thorough digital reconnaissance on the provided {topic}. Gather all public information, including links, social media profiles, and text snippets.',
    expected_output='A raw, unorganized list containing all the data points, links, and text snippets found during the investigation.',
    agent=recon_agent
)

# Task 2: Analyze and Cluster Data
analysis_task = Task(
    description='Analyze the raw data from the reconnaissance phase. Cluster the information into logical groups to create a structured markdown report.',
    expected_output='A well-structured markdown report with clear headings (e.g., ## Professional Information) and bullet points listing the relevant data.',
    agent=analyst_agent
)

# Task 3: Assess Security Risks
security_task = Task(
    description='Conduct a final security risk assessment based on the structured report. Assign a risk score, provide a justification for the score, and list actionable recommendations.',
    expected_output='A final report containing a numerical risk score (1-10), a paragraph explaining the reasoning behind the score, and a bulleted list of 3-5 actionable steps the subject can take to improve their digital security.',
    agent=security_agent
)

# --- CREW DEFINITION ---
# Create the crew with the defined agents and tasks.
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
    "This application leverages the CrewAI framework and a HuggingFace language model (Mistral-7B) "
    "to perform a multi-step analysis of a subject's publicly available digital footprint. The process is broken down into three phases: reconnaissance, analysis, and security assessment."
)

st.header("Enter a Target for Reconnaissance")
topic_input = st.text_input(
    "Enter a name, username, or email address to investigate:",
    placeholder="e.g., 'John Doe', 'johndoe123', or 'john.doe@email.com'"
)

if st.button("🕵️‍♂️ Start Analysis", type="primary"):
    if topic_input:
        with st.spinner("The AI crew is starting the investigation... This may take a few minutes depending on the complexity."):
            st.info("🔄 **Crew Kickoff:** The investigation has started.")
            
            # Add placeholders for agent updates
            recon_status = st.empty()
            analysis_status = st.empty()
            security_status = st.empty()
            
            try:
                # The kickoff method starts the crew's process.
                # The inputs dictionary passes the user's topic to the tasks.
                result = digital_footprint_crew.kickoff(inputs={'topic': topic_input})
                
                st.success("Analysis Complete!")
                st.markdown("---")
                st.header("Final Digital Footprint Report")
                st.markdown(result)
                
            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")
    else:
        st.warning("Please enter a target to investigate.")

