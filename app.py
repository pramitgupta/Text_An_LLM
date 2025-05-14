import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from wordcloud import WordCloud
import nltk
import tempfile
import os

nltk.download('punkt')
nltk.download('stopwords')

# Load NRC
def load_nrc(filepath="/content/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"): #you need to download that and save in to your colab path
    df = pd.read_csv(filepath, sep='\t', header=None)
    df.columns = ['word', 'emotion', 'association']
    return df[df['association'] == 1]

nrc_df = load_nrc()

# Main analysis function
def analyze_and_plot(file, id_col, text_col, stopword_input):
    df = pd.read_csv(file.name)
    stop_words = set(stopwords.words('english'))
    if stopword_input:
        stop_words.update([s.strip() for s in stopword_input.split(',')])

    emotion_records = []
    emotion_words = defaultdict(list)

    for _, row in df.iterrows():
        row_id = row[id_col]
        text = str(row[text_col])
        sentences = sent_tokenize(text)

        for sent in sentences:
            words = word_tokenize(sent.lower())
            words = [w for w in words if w.isalpha() and w not in stop_words]
            matched = nrc_df[nrc_df['word'].isin(words)]
            for _, r in matched.iterrows():
                emotion_records.append({'ID': row_id, 'emotion': r['emotion']})
                emotion_words[r['emotion']].append(r['word'])

    emotion_df = pd.DataFrame(emotion_records)

    # Emotion plot
    g = sns.FacetGrid(emotion_df, col='emotion', col_wrap=2, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x="ID", bins=len(emotion_df['ID'].unique()), discrete=True)
    g.set_titles("{col_name}")
    g.set_axis_labels("ID", "Count")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("NRC Emotion Distribution per Row ID")
    facet_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    g.savefig(facet_path)
    plt.close()

    # Word clouds
    wc_paths = []
    for emotion, words in emotion_words.items():
        freq = Counter(words)
        wc = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(freq)
        fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.set_title(f"Word Cloud: {emotion}")
        ax_wc.axis('off')
        cloud_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        fig_wc.savefig(cloud_path)
        plt.close(fig_wc)
        wc_paths.append(cloud_path)

    return facet_path, wc_paths

# Populate column dropdowns
def get_column_names(file):
    df = pd.read_csv(file.name)
    columns = list(df.columns)
    return gr.update(choices=columns), gr.update(choices=columns)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## NRC Sentiment Analysis App with Word Clouds")

    file_input = gr.File(label="Upload CSV File")

    with gr.Row():
        id_col = gr.Dropdown(label="Select ID Column")
        text_col = gr.Dropdown(label="Select Text Column")
    stopword_input = gr.Textbox(label="Stop Words (comma-separated)", placeholder="e.g. will,can")

    run_btn = gr.Button("Analyze")

    with gr.Row():
        plot_output = gr.Image(type="filepath", label="Sentiment Plot")
        wordcloud_gallery = gr.Gallery(label="Word Clouds", columns=2, object_fit="contain")

    file_input.change(fn=get_column_names, inputs=[file_input], outputs=[id_col, text_col])
    run_btn.click(fn=analyze_and_plot,
                  inputs=[file_input, id_col, text_col, stopword_input],
                  outputs=[plot_output, wordcloud_gallery])

demo.launch(share=True)
