import spaces  
import gradio as gr
import pandas as pd
import os
from datetime import datetime

# Важно для серверов Hugging Face (отрисовка графиков без монитора)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# Загрузка ИИ-модели
model = SentenceTransformer('all-MiniLM-L6-v2')
DB_FILE = "dream_database.csv"

# Инициализация базы данных
def init_db():
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["Timestamp", "Alias", "ASC_Type", "Emotion", "Intensity", "Narrative", "Vector_Preview"])
        df.to_csv(DB_FILE, index=False)

init_db()

# --- ФУНКЦИИ ДЛЯ СБОРА ДАННЫХ (Вкладка 1) ---
@spaces.GPU
def get_embedding(text):
    return model.encode(text)

@spaces.GPU
def get_embeddings_bulk(texts):
    return model.encode(texts)

def process_entry(alias, asc_type, emotion, intensity, narrative):
    if not narrative.strip():
        return "Error: Please describe your experience.", "", pd.read_csv(DB_FILE).tail(5)
    
    embedding = get_embedding(narrative)
    vector_preview = f"[{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, {embedding[3]:.4f}, {embedding[4]:.4f} ... 384 dimensions]"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[timestamp, alias, asc_type, emotion, intensity, narrative, vector_preview]], 
                            columns=["Timestamp", "Alias", "ASC_Type", "Emotion", "Intensity", "Narrative", "Vector_Preview"])
    
    new_data.to_csv(DB_FILE, mode='a', header=False, index=False)
    updated_df = pd.read_csv(DB_FILE)
    
    success_msg = f"Thank you, {alias}! Your experience has been digitized."
    return success_msg, vector_preview, updated_df.tail(10)

# --- ФУНКЦИИ ДЛЯ АНАЛИЗА ДАННЫХ (Вкладка 2) ---
def analyze_database():
    df = pd.read_csv(DB_FILE)
    df = df.dropna(subset=['Narrative'])
    
    # Защита от слишком малого количества данных
    if len(df) < 4:
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, f'Need at least 4 entries for AI clustering.\nCurrently: {len(df)} entries.', 
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        return fig, df
        
    # Векторизация всех текстов на видеокарте
    texts = df['Narrative'].tolist()
    embeddings = get_embeddings_bulk(texts)
    
    # Кластеризация (динамический расчет групп)
    num_clusters = min(3, max(2, len(df) // 3))
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(embeddings)
    
    # Сжатие до 2D (t-SNE)
    perplexity = min(5, len(df) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(embeddings)
    
    df['X'] = vectors_2d[:, 0]
    df['Y'] = vectors_2d[:, 1]
    
    # Отрисовка графика
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='X', y='Y',
        hue='Cluster',
        style='Emotion',
        size='Intensity',
        sizes=(50, 200),
        palette='viridis',
        data=df
    )
    
    plt.title("DreamCode Semantic Map: ASC Vector Projections", fontsize=14)
    plt.xlabel("Semantic Dimension 1")
    plt.ylabel("Semantic Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Возвращаем график и таблицу с кластерами (без координат X,Y для чистоты)
    display_df = df[['Timestamp', 'Alias', 'Emotion', 'Cluster', 'Narrative']]
    return fig, display_df

# ----------------- ИНТЕРФЕЙС GRADIO -----------------
with gr.Blocks() as app:
    gr.Markdown("# 🌌 DreamCode: ASC Research Platform")
    
    with gr.Tabs():
        # ВКЛАДКА 1: СБОР ДАННЫХ
        with gr.TabItem("1. Data Ingestion"):
            gr.Markdown("Submit your Altered State of Consciousness (ASC) experiences here.")
            with gr.Row():
                with gr.Column():
                    alias = gr.Textbox(label="Alias / Participant ID")
                    asc_type = gr.Dropdown(choices=["Ordinary Dream", "Lucid Dream (LD)", "OBE", "NDE", "Other"], label="State")
                    emotion = gr.Radio(choices=["Positive", "Neutral", "Negative"], label="Emotional Tone")
                    intensity = gr.Slider(minimum=1, maximum=3, step=1, label="Intensity (1-3)")
                    narrative = gr.Textbox(label="Narrative", lines=5)
                    submit_btn = gr.Button("Submit & Analyze", variant="primary")
                    
                with gr.Column():
                    status_output = gr.Textbox(label="Status", interactive=False)
                    vector_output = gr.Textbox(label="Vector Generation (Preview)", interactive=False)
                    data_preview = gr.Dataframe(headers=["Timestamp", "Alias", "ASC_Type", "Emotion", "Intensity", "Narrative", "Vector_Preview"], interactive=False)

            submit_btn.click(
                fn=process_entry, 
                inputs=[alias, asc_type, emotion, intensity, narrative], 
                outputs=[status_output, vector_output, data_preview]
            )

        # ВКЛАДКА 2: АНАЛИТИКА (DASHBOARD)
        with gr.TabItem("2. AI Analysis Dashboard"):
            gr.Markdown("### Mathematical Clustering of ASC Narratives\nVisualize the hidden semantic connections between global anomalous experiences.")
            analyze_btn = gr.Button("Generate AI Semantic Map (t-SNE & K-Means)", variant="primary")
            
            with gr.Row():
                plot_output = gr.Plot(label="Semantic Vector Map")
            
            gr.Markdown("### Data Grouped by AI Clusters")
            cluster_data = gr.Dataframe(interactive=False)
            
            analyze_btn.click(
                fn=analyze_database,
                inputs=[],
                outputs=[plot_output, cluster_data]
            )

app.launch(theme=gr.themes.Monochrome())