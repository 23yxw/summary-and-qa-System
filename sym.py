from transformers import BartForConditionalGeneration, BartTokenizer
import requests
from bs4 import BeautifulSoup
import nltk
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import os
from math import log
import contractions
import gradio as gr
import re
from pdfdocument.document import PDFDocument
from google.cloud import storage


nltk.download('punkt')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "voltaic-psyche-394213-aa9b011a5b38.json"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
llm2 = HuggingFaceHub(repo_id="wyx-ucl/bart-EDGAR-CORPUS", model_kwargs={"max_length": 200})
openai.api_key = "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"
os.environ['OPENAI_API_KEY'] = "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"
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="text-davinci-003")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
chain2 = load_qa_chain(llm, chain_type="stuff", verbose=False)
# separate words that stick together
# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("5.txt").read().split()
wordcost = dict((k, log((i + 1) * log(len(words)))) for i, k in enumerate(words))
maxword = max(len(x) for x in words)


def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i - maxword):i]))
        return min((c + wordcost.get(s[i - k - 1:i], 9e999), k + 1) for k, c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1, len(s) + 1):
        c, k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i > 0:
        c, k = best_match(i)
        assert c == cost[i]
        out.append(s[i - k:i])
        i -= k

    return " ".join(reversed(out))

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Pre-processing of the input text into chunks
def preprocess_text(text):
    # Segmenting Text with NLTK's Sentence Segmenter
    sentences = nltk.sent_tokenize(text)
    chunks = []
    chunk = ""
    for sentence in sentences:
        temp_chunk = chunk + " " + sentence if chunk else sentence
        if len(tokenizer.tokenize(temp_chunk)) <= 980:
            chunk = temp_chunk
        else:
            chunks.append(chunk)
            chunk = sentence
    if chunk:
        chunks.append(chunk)
    return chunks


def summarize(text):
    chunks = preprocess_text(text)
    summaries = []
    for chunk in chunks:
        # Generate a summary for each chunk
        inputs = tokenizer.encode(chunk, return_tensors='pt')
        summary_ids = model.generate(inputs, num_beams=2, min_length=0, max_length=150)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Stringing together the generated summaries
    return " ".join(summaries)

def summarize_with_gpt3(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
    response = chain.run(docs)
    return response

def scrape_data(url,mo,file_type):
    if not url:
        output="please input a valid URL"
        link ="please input a valid URL"
        return output,link
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup.find_all("figure"):
            tag.clear()
        article_Body = " "
        body = soup.article

        for data in body.find_all("p"):
            article_Body += data.get_text()


        # Text Cleaning
        z = nltk.sent_tokenize(article_Body)
        processed_sentences = []
        processed_sentences2 = []
        arr2 = []
        for sentence in z:
            sentence = re.sub(r'-[\n\r]+', "", sentence)
            sentence = re.sub(r'-', ' ', sentence)
            sentence = re.sub(r'\n', ' ', sentence)
            sentence = contractions.fix(sentence)
            sentence = re.sub(r'\([^()]*\)', '', sentence)
            sentence = sentence.lower()
            processed_sentences.append(sentence)
        for s in processed_sentences:
            m = re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                          s)
            n = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', s)
            if m:
                pass
            elif n:
                pass
            else:
                arr2.append(s)
        for sen in arr2:
            words_list = nltk.word_tokenize(sen)
            words_list2 = []

            for word in words_list:
                if len(word) > 12:
                    w = infer_spaces(word)
                    u = w.split()
                    for q in u:
                        words_list2.append(q)
                else:
                    words_list2.append(word)

            recombined_string = ' '.join(words_list2)

            processed_sentences2.append(recombined_string)
        article = ' '.join(processed_sentences2)
        if "BART" in mo and len(mo) == 1:
            output = summarize(article)
            if 'txt' in file_type and len(file_type) == 1:
                le = str(hash(url))
                typ = str(file_type[0])
                filename = "summary_bart" + le + "." + typ
                with open(filename, 'w') as f:
                    f.write(output)
                    # Create a storage client
                storage_client = storage.Client()

                # Get the bucket that the file will be uploaded to
                bucket = storage_client.get_bucket('0073')

                # Create a new blob and upload the file's content
                blob = bucket.blob(filename)
                blob.upload_from_filename(filename)

                # Make the blob publicly viewable
                blob.make_public()
                link = blob.public_url

            elif 'pdf' in file_type and len(file_type) == 1:
                le = str(hash(url))
                typ = str(file_type[0])
                filename = "summary_bart" + le + "." + typ
                pdf = PDFDocument(filename)
                pdf.init_report()
                pdf.h1('Summary')
                pdf.p(output)
                pdf.generate()
                # Create a storage client
                storage_client = storage.Client()

                # Get the bucket that the file will be uploaded to
                bucket = storage_client.get_bucket('0073')

                # Create a new blob and upload the file's content
                blob = bucket.blob(filename)
                blob.upload_from_filename(filename)

                # Make the blob publicly viewable
                blob.make_public()
                link = blob.public_url

            else:
                link = "Invalid file type. Choose from 'txt', 'pdf'."

        elif "GPT-3.5" in mo and len(mo) == 1:
            output = summarize_with_gpt3(article)
            if 'txt' in file_type and len(file_type) == 1:
                le = str(hash(url))
                typ = str(file_type[0])
                filename = "summary_gpt" + le + "." + typ
                with open(filename, 'w') as f:
                    f.write(output)
                    # Create a storage client
                storage_client = storage.Client()

                # Get the bucket that the file will be uploaded to
                bucket = storage_client.get_bucket('0073')

                # Create a new blob and upload the file's content
                blob = bucket.blob(filename)
                blob.upload_from_filename(filename)

                # Make the blob publicly viewable
                blob.make_public()
                link = blob.public_url

            elif 'pdf' in file_type and len(file_type) == 1:
                le = str(hash(url))
                typ = str(file_type[0])
                filename = "summary_gpt" + le + "." + typ
                pdf = PDFDocument(filename)
                pdf.init_report()
                pdf.h1('Summary')
                pdf.p(output)
                pdf.generate()
                # Create a storage client
                storage_client = storage.Client()

                # Get the bucket that the file will be uploaded to
                bucket = storage_client.get_bucket('0073')

                # Create a new blob and upload the file's content
                blob = bucket.blob(filename)
                blob.upload_from_filename(filename)

                # Make the blob publicly viewable
                blob.make_public()
                link = blob.public_url

            else:
                link = "Invalid file type. Choose from 'txt', 'pdf'."

        else:
            output = "Error: Invalid model choice"
            link = "No files will be generated due to invalid model choice"

        return output,link
    except:
        link = "please input a valid URL"
        output = "please input a valid URL"
        return output, link


# Interface for summarizing articles from news sites
iface1 = gr.Interface(
    fn=scrape_data,
    inputs=[gr.Textbox(label="Website address"), gr.CheckboxGroup(
        ["BART", "GPT-3.5"], label="Select a model"),gr.CheckboxGroup(
        ['txt', 'pdf'], label="Select a format of the summary report")], outputs=[gr.Textbox(),gr.Textbox()])


def summarize_docs2(docs):
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=980, chunk_overlap=20)
    split_docs = text_splitter.split_documents(docs)
    w = []
    for doc in split_docs:
        page_contents = doc.page_content
        v = llm2.predict(page_contents)
        w.append(v)
    return " ".join(w)


def summarize_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
    response = chain.run(input_documents=split_docs)
    return response


def extract_text_from_pdf(file, mod,file_type, startpage, endpage,):
    if file is None:
        output2="Please upload a file"
        link="Please upload a file"
        return output2,link

    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()
    s = startpage - 1
    e = endpage
    l = len(pages)
    if startpage <= endpage <= l:
        if "BART" in mod and len(mod) == 1:
            output2 = summarize_docs2(pages[s:e])
            if 'txt' in file_type and len(file_type) == 1:
                name = str(file.name)
                name = name.replace('.pdf', '')
                le = str(hash(name))
                ty = str(file_type[0])
                filename = "summary_pdf_bart_" + le + "." + ty
                with open(filename, 'w') as f:
                    f.write(output2)
                    # Create a storage client
                storage_client = storage.Client()

                # Get the bucket that the file will be uploaded to
                bucket = storage_client.get_bucket('0073')

                # Create a new blob and upload the file's content
                blob = bucket.blob(filename)
                blob.upload_from_filename(filename)

                # Make the blob publicly viewable
                blob.make_public()
                link = blob.public_url

            elif 'pdf' in file_type and len(file_type) == 1:
                name = str(file.name)
                name = name.replace('.pdf', '')
                le = str(hash(name))
                ty = str(file_type[0])
                filename = "summary_pdf_bart_" + le + "." + ty
                pdf = PDFDocument(filename)
                pdf.init_report()
                pdf.h1('Summary')
                pdf.p(output2)
                pdf.generate()
                # Create a storage client
                storage_client = storage.Client()

                # Get the bucket that the file will be uploaded to
                bucket = storage_client.get_bucket('0073')

                # Create a new blob and upload the file's content
                blob = bucket.blob(filename)
                blob.upload_from_filename(filename)

                # Make the blob publicly viewable
                blob.make_public()
                link = blob.public_url


            else:
                link = "Invalid file type. Choose from 'txt', 'pdf'."
        elif "GPT-3.5" in mod and len(mod) == 1:
            output2 = summarize_docs(pages[s:e])
            if 'txt' in file_type and len(file_type) == 1:
                name = str(file.name)
                name = name.replace('.pdf', '')
                le = str(hash(name))
                ty = str(file_type[0])
                filename = "summary_pdf_gpt_" + le + "." + ty
                with open(filename, 'w') as f:
                    f.write(output2)
                    # Create a storage client
                storage_client = storage.Client()

                # Get the bucket that the file will be uploaded to
                bucket = storage_client.get_bucket('0073')

                # Create a new blob and upload the file's content
                blob = bucket.blob(filename)
                blob.upload_from_filename(filename)

                # Make the blob publicly viewable
                blob.make_public()
                link = blob.public_url

            elif 'pdf' in file_type and len(file_type) == 1:
                name = str(file.name)
                name = name.replace('.pdf', '')
                le = str(hash(name))
                ty = str(file_type[0])
                filename = "summary_pdf_gpt_" + le + "." + ty
                pdf = PDFDocument(filename)
                pdf.init_report()
                pdf.h1('Summary')
                pdf.p(output2)
                pdf.generate()
                # Create a storage client
                storage_client = storage.Client()

                # Get the bucket that the file will be uploaded to
                bucket = storage_client.get_bucket('0073')

                # Create a new blob and upload the file's content
                blob = bucket.blob(filename)
                blob.upload_from_filename(filename)

                # Make the blob publicly viewable
                blob.make_public()
                link = blob.public_url


            else:
                link = "Invalid file type. Choose from 'txt', 'pdf'."
        else:
            output2 = "Error: Invalid model choice"
            link = "No files will be generated due to invalid model choice"



        return output2, link

    else:
        return "please make sure: s less than or equal to e ", "please make sure: s less than or equal to e "



# Interface for summarising PDF documents (e.g. annual reports)
iface2 = gr.Interface(
    fn=extract_text_from_pdf,
    inputs=[gr.File(), gr.CheckboxGroup(
        ["BART", "GPT-3.5"], label="the language model"), gr.CheckboxGroup(
        ['txt', 'pdf'], label="Select a format of the summary report"), gr.Slider(minimum=1, maximum=300, step=1, label="summarise from page: s"),
            gr.Slider(minimum=1, maximum=300, step=1, label="to page: e")],
    outputs=[gr.Textbox(),gr.Textbox()])


def chat(file, prompt):
    if file is None:
        return "Please upload a file","Please upload a file"
    if not prompt:
        return "Please enter a question","Please enter a question"
    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(pages)
    vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name='annualreport')
    docs = vectorstore.similarity_search(prompt, 3, include_metadata=True)
    re = chain2.run(input_documents=docs, question=prompt)
    first_page_content = docs[0].page_content
    first_source = docs[0].metadata['page']
    page_num = str(first_source)
    si = first_page_content + '\npage:' + page_num
    return re, si


iface3 = gr.Interface(
    fn=chat,
    inputs=[gr.File(), gr.Textbox(label="Input your prompt here")],
    outputs=[gr.Textbox(label='Answer'), gr.Textbox(label='Document Similarity Search')])

demo = gr.TabbedInterface([iface1, iface2, iface3],
                          ["Summarise financial news article from a news website", "Summarise the PDF file",
                           "Chat with the PDF file"])

demo.launch(share=True)