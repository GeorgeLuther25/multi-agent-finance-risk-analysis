import streamlit as st
from openai import OpenAI
import os

st.set_page_config(page_title="ChatGPT UI", page_icon="🤖", layout="centered")

st.title("ChatGPT-like UI with OpenAI API")

# api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
# if api_key:
#     openai.api_key = api_key
# else:
#     st.warning("Please enter your OpenAI API key in the sidebar.")
#     st.stop()

api_key = 'add-your-api-key-here'
client = OpenAI(api_key=api_key)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
user_input = st.text_area("Your message:", value="", height=100, key="user_input")
submit = st.button("Send", type="primary")

if submit and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo",
            model="gpt-5",
            messages=st.session_state.chat_history,
            # temperature=0.7,
            # max_tokens=512,
        )
        assistant_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("## ChatGPT Output")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**ChatGPT:** {msg['content']}")
