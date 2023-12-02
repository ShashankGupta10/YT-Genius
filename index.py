from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.llms.cohere import Cohere
from langchain.memory import ConversationBufferMemory
import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# Title and Input Fields
st.title("Youtube Script Generator 🦜🔗")
prompt_topic = st.text_input("Enter your topic here")


# Prompt templates
title_template = PromptTemplate(
    input_variables=["topic"],
    template="Give me an attractive youtube video title about {topic}.\
    The format of the output should be just the title and not anything else.\
    Don't even give a complete sentence just give the title.\
    Give only one title."
)

script_template = PromptTemplate(
    input_variables=["title"],
    template="Generate me a youtube video script about the title {title}.\
    It should be very energetic and should connect with the audience.\
    The format of the output should be just the script and not anything else.\
    Don't give any greetings just the script.\
    The script should have an Introduction, a body and an outro.\
    Give 100 words at least"
)

# Memory
memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")

# LLM Initialization
llm = Cohere(temperature=0.9, cohere_api_key=st.secrets["COHERE_API_KEY"])

# LLM Chains

# Title Chain
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=memory)
# Script Chain
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script", memory=memory)

# Sequential chain for communication between both chains
simple_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=["topic"], output_variables=["title", "script"], verbose=True)

# Output to the user
if (prompt_topic):
    response = simple_chain({"topic": prompt_topic})
    print(response)
    st.subheader("Title:")
    st.write(response["title"])
    st.subheader("Script:")
    st.write(response["script"])

    with st.expander("Message History:"):
        st.info(memory.buffer)
