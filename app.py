import streamlit as st
from transformers import pipeline

# Try to import OpenAI (optional, if user provides API key)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Default Hugging Face model (free, no key needed)
hf_generator = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_with_hf(prompt, max_tokens=200):
    return hf_generator(prompt, max_new_tokens=max_tokens)[0]["generated_text"]

def generate_with_openai(prompt, api_key, max_tokens=200):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# ----- Employee AI -----
class EmployeeAI:
    def __init__(self, role, api_key=None):
        self.role = role
        self.api_key = api_key

    def work(self, task):
        prompt = f"You are an AI {self.role}. Complete this task:\n{task}"
        if self.api_key and OpenAI:
            return generate_with_openai(prompt, self.api_key, 150)
        return generate_with_hf(prompt, 150)

# ----- Project Manager AI -----
class ProjectManagerAI:
    def __init__(self, employees, api_key=None):
        self.employees = employees
        self.api_key = api_key

    def breakdown_project(self, project_description):
        prompt = f"Break down this project into 3 clear tasks:\n{project_description}"
        if self.api_key and OpenAI:
            text = generate_with_openai(prompt, self.api_key, 100)
        else:
            text = generate_with_hf(prompt, 100)
        return [t.strip("-• ") for t in text.split("\n") if t.strip()]

    def assign_and_collect(self, tasks):
        results = []
        for task, employee in zip(tasks, self.employees):
            result = employee.work(task)
            results.append((employee.role, task, result))
        return results

    def combine_results(self, results):
        combined_text = "\n\n".join(
            [f"{role} ({task}): {res}" for role, task, res in results]
        )
        prompt = f"Combine the following work into a polished final output:\n{combined_text}"
        if self.api_key and OpenAI:
            return generate_with_openai(prompt, self.api_key, 200)
        return generate_with_hf(prompt, 200)

# ----- Streamlit UI -----
st.set_page_config(page_title=" Sync Space", layout="wide")

st.title("Sync Space")
st.write("An AI Project Manager assigns tasks to AI Employees, collects results, and produces a final deliverable.")

# User API key input
api_key = st.text_input("Enter your OpenAI API Key (optional)", type="password")

project_description = st.text_area("Enter your project description:", height=100)

if st.button("Run Project"):
    if project_description.strip():
        employees = [
            EmployeeAI("Researcher", api_key),
            EmployeeAI("Writer", api_key),
            EmployeeAI("Editor", api_key),
        ]
        manager = ProjectManagerAI(employees, api_key)

        st.subheader("📋 Task Breakdown")
        tasks = manager.breakdown_project(project_description)
        for i, task in enumerate(tasks):
            st.write(f"**Task {i+1}:** {task}")

        st.subheader("👷 Employee Outputs")
        results = manager.assign_and_collect(tasks)
        for role, task, output in results:
            with st.expander(f"{role} - {task}"):
                st.write(output)

        st.subheader("✅ Final Output")
        final_output = manager.combine_results(results)
        st.success(final_output)
    else:
        st.warning("Please enter a project description.")
