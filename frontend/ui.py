import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/ask-rag"


st.set_page_config(
    page_title="Enterprise AI Knowledge Copilot",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 Enterprise AI Knowledge Copilot")

st.write(
    "請輸入企業文件相關問題，"
    "AI 將從內部知識庫搜尋資料後回答。"
)


question = st.text_area(
    "你的問題",
    placeholder="例如：送測需要幾台樣品？",
    height=100
)


if st.button(
    "詢問 AI",
    type="primary"
):

    if not question.strip():

        st.warning(
            "請先輸入問題。"
        )

    else:

        with st.spinner(
            "正在搜尋企業知識庫..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question
                    },
                    timeout=60
                )

                if response.status_code == 200:

                    data = response.json()

                    st.subheader("AI 回答")

                    st.write(
                        data["answer"]
                    )

                    st.subheader("參考來源")

                    for source in data["sources"]:

                        with st.expander(
                            f"{source['source']} "
                            f"｜Chunk {source['chunk_id']}"
                        ):

                            st.write(
                                "相關度：",
                                source["score"]
                            )

                else:

                    st.error(
                        f"API 錯誤："
                        f"{response.status_code}"
                    )

                    st.write(
                        response.text
                    )

            except requests.exceptions.RequestException as e:

                st.error(
                    "無法連線到 AI Backend。"
                )

                st.write(
                    str(e)
                )