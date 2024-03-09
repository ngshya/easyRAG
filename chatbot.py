import dspy
import time
import streamlit as st


lm = dspy.HFModel(model="google/flan-t5-base", )
retriever = dspy.ColBERTv2(url='http://localhost:8001/api/search')
dspy.settings.configure(lm=lm, rm=retriever)

class BasicQA(dspy.Signature):
    """Answer the question with a short answer."""

    question = dspy.InputField()
    answer = dspy.OutputField()

qa = dspy.Predict(BasicQA)

class GenerateAnswerWithContext(dspy.Signature):
    """Answer the question with a short answer."""

    context = dspy.InputField(desc="may contain relevant facts to consider")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="short answer")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswerWithContext)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    
rag = RAG()



st.title('Fringe Chatbot')
st.sidebar.title("Chat History")

task = st.sidebar.radio(
    "Simple QA or RAG?",
    ["Simple QA", "RAG"]
)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

txt = st.chat_input("Say something...")
if txt:
    st.session_state['chat_history'].append("User: "+txt)
    chat_user = st.chat_message("user")
    chat_user.write(txt)
    chat_assistant = st.chat_message("assistant")
    with st.status("Generating the answer...") as status:
        tms_start = time.time()
        if task == "Simple QA":
            ans = qa(question=txt).answer
        elif task == "RAG":
            ans = rag(question=txt).answer
        chat_assistant.write(ans)
        st.session_state['chat_history'].append("Assistant: "+ans)
        tms_elapsed = time.time() - tms_start
        status.update(label="Answer generated in %0.2f seconds." % (tms_elapsed), state="complete", expanded=False)
    st.sidebar.markdown("<br />".join(st.session_state['chat_history'])+"<br /><br />", unsafe_allow_html=True)
