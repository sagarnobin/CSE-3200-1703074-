import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model_checkpoint = r"C:\Users\HP\Desktop\Project_3200\pegasus-samsum-model"
tokenizer = r"C:\Users\HP\Desktop\Project_3200\tokenizer"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


def main():
    st.title('Text Summarization')

    # Input text box
    st.header("Enter Your Text:")
    input_text = st.text_area("Input Text", "", height=200)

    # Summarize button
    if st.button("Summarize"):
        if input_text:
            # Perform summarization
            summary = generate_summary(input_text)
            # Display the generated summary
            st.header("Summarization:")
            st.text_area("Generated Summary", summary, height=200)
        else:
            st.warning("Please enter text before summarizing.")


# Function to generate summary using the fine-tuned model
def generate_summary(text):
    summary = summarizer(text, max_length=150, min_length=50, length_penalty=5.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']

if __name__ == "__main__":
    main()
