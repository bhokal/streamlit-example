import streamlit as st
from langchain.llms import HuggingFacePipeline
from bedrock_client import BedrocClient

# Initialize the Bedrock client
bedrock_client = BedrocClient()

# Load the LLM from Bedrock
llm = HuggingFacePipeline.from_bedrock(bedrock_client, model_name="your_model_name")

def run_ai_solution_design_process(initial_input):
    # Step 1: Persona creation and discussions
    step1_input = f"""
    <initial_input>
    {initial_input}
    </initial_input>

    <step1>
    <input>
    &lt;initial_input&gt;{initial_input}&lt;/initial_input&gt;
    </input>
    Create 3 personas: 1. product manager, 2. architect/developer, 3. design reviewer. Have two discussions: business requirement discussion (between product manager and architect/developer) and architecture discussion (between architect/developer and design reviewer).
    <output>
    """

    step1_output = llm(step1_input)

    step1_output += """
    </output>
    </step1>
    """

    # Step 2: BRD discussion and requirements
    step2_input = f"""
    <input>
    &lt;initial_input&gt;{initial_input}&lt;/initial_input&gt;
    &lt;previous_output&gt;{step1_output}&lt;/previous_output&gt;
    </input>
    Have a BRD (business requirements document) discussion for the proposed AI solution, involving the product manager and architect/developer personas. Summarize the discussion and come up with functional and non-functional requirements.
    <output>
    """

    step2_output = llm(step2_input)

    step2_output += """
    </output>
    """

    # Step 3: Architecture discussion and technical design document
    step3_input = f"""
    <input>
    &lt;initial_input&gt;{initial_input}&lt;/initial_input&gt;
    &lt;previous_output&gt;{step2_output}&lt;/previous_output&gt;
    </input>
    Involve an AWS cloud expert and have an architecture discussion, focusing on high-level components, their interactions, and the relevant AWS cloud services. Capture the entire design in a single technical document response.
    <output>
    """

    step3_output = llm(step3_input)

    step3_output += """
    </output>
    """

    # Step 4: Metrics, alarms, and dashboard strategy
    step4_input = f"""
    <input>
    &lt;initial_input&gt;{initial_input}&lt;/initial_input&gt;
    &lt;previous_output&gt;{step3_output}&lt;/previous_output&gt;
    </input>
    Provide a metrics, alarms, and dashboard strategy for monitoring and alerting in the proposed AI solution.
    <output>
    """

    step4_output = llm(step4_input)

    step4_output += """
    </output>
    """

    # Step 5: Cloud architect review, trade-offs, pros, and cons
    step5_input = f"""
    <input>
    &lt;initial_input&gt;{initial_input}&lt;/initial_input&gt;
    &lt;previous_output&gt;{step3_output}&lt;/previous_output&gt;
    &lt;previous_output&gt;{step4_output}&lt;/previous_output&gt;
    </input>
    Involve a cloud architect to review the architectural design document and the metrics, alarms, and dashboard strategy. Identify the top five areas of trade-offs and discuss their pros and cons.
    <output>
    """

    step5_output = llm(step5_input)

    step5_output += """
    </output>
    """

    return step1_output, step2_output, step3_output, step4_output, step5_output

# Streamlit app
st.title("AI Solution Design Process")

brd_input = st.text_area("Enter the Business Requirements Document (BRD) for the AI solution:")

if brd_input:
    with st.spinner("Running the AI Solution Design Process..."):
        step1_output, step2_output, step3_output, step4_output, step5_output = run_ai_solution_design_process(brd_input)

    st.header("Step 1: Persona Creation and Discussions")
    st.write(step1_output)

    st.header("Step 2: BRD Discussion and Requirements")
    st.write(step2_output)

    st.header("Step 3: Architecture Discussion and Technical Design Document")
    st.write(step3_output)

    st.header("Step 4: Metrics, Alarms, and Dashboard Strategy")
    st.write(step4_output)

    st.header("Step 5: Cloud Architect Review, Trade-offs, Pros, and Cons")
    st.write(step5_output)
