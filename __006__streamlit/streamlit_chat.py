import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

import streamlit as st
import requests
from langgraph_workflow import call_langgraph_ai

st.set_page_config(page_title="中医知识图谱问答", page_icon=" 中医")
st.title(" 中医知识图谱智能问答系统")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("请输入您的中医问题..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    try:
        is_xhs, reply = call_langgraph_ai("user_001", "session_001", user_input)
        if is_xhs:
            title, content, image_path = reply
            with st.chat_message("assistant"):
                st.markdown(f"**{title}**")
                st.markdown(content)
                if image_path and os.path.exists(image_path):
                    st.image(image_path)
                st.success("内容已准备就绪，可手动发布到小红书")
        else:
            response = reply[0]
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"发生错误: {e}")
