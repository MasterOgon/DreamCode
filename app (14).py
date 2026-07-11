import spaces  
import gradio as gr
import pandas as pd
import numpy as np
import os
import re

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ ИИ
# ==========================================
model = SentenceTransformer('all-MiniLM-L6-v2')

def split_into_sentences(text):
    # Разбиваем по точкам, восклицательным и вопросительным знакам
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 10]

# ==========================================
# 2. КОНФИГУРАЦИЯ БАЗЫ ДАННЫХ
# ==========================================
DATASET_REPO_ID = "Masterogon/dream-database" 
DB_FILE = "dream_database.csv"
HF_TOKEN = os.environ.get("HF_TOKEN")

api = HfApi()

def init_db():
    if HF_TOKEN:
        try:
            print("Downloading database from Hugging Face Hub...")
            file_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename=DB_FILE, repo_type="dataset", token=HF_TOKEN)
            import shutil
            shutil.copy(file_path, DB_FILE)
            print("Database successfully loaded.")
            return
        except Exception as e:
            print("Could not download DB. It might be empty or missing. Error:", e)
    
    if not os.path.exists(DB_FILE):
        df = pd.DataFrame(columns=["Timestamp", "Alias", "ASC_Type", "Emotion", "Intensity", "Narrative"])
        df.to_csv(DB_FILE, index=False)
        print("Created a fresh local database.")

init_db()

# ==========================================
# 3. ФУНКЦИЯ ЗАПИСИ (Data Ingestion)
# ==========================================
def process_entry(alias, asc_type, emotion, intensity, narrative):
    if not narrative.strip():
        return "Error: Please describe your experience.", pd.read_csv(DB_FILE).tail(5)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[timestamp, alias, asc_type, emotion, intensity, narrative]], 
                            columns=["Timestamp", "Alias", "ASC_Type", "Emotion", "Intensity", "Narrative"])
    new_data.to_csv(DB_FILE, mode='a', header=False, index=False)
    
    if HF_TOKEN:
        try:
            api.upload_file(
                path_or_fileobj=DB_FILE,
                path_in_repo=DB_FILE,
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Added new report by {alias}"
            )
            backup_status = "Successfully synced to cloud."
        except Exception as e:
            backup_status = f"Warning: Cloud sync failed ({e})"
    else:
        backup_status = "Warning: No HF_TOKEN. Data is local."
        
    return f"Success! Report added. {backup_status}", pd.read_csv(DB_FILE).tail(10)

# ==========================================
# 4. ФУНКЦИЯ ДЛЯ ПРОСМОТРА ПОЛНОЙ БАЗЫ (Tab 5)
# ==========================================
def view_database():
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    df = pd.read_csv(DB_FILE).dropna(subset=['Narrative']).reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame()
    
    # Создаем анонимные ID для удобного чтения
    df.insert(0, 'Report_ID', [f"Report #{i+1}" for i in range(len(df))])
    # Возвращаем нужные столбцы, скрывая реальные имена (Alias)
    return df[['Report_ID', 'Timestamp', 'ASC_Type', 'Emotion', 'Intensity', 'Narrative']]

# ==========================================
# 5. ФУНКЦИИ МАКРО-АНАЛИЗА (Tab 2: Heatmap & Graph)
# ==========================================
@spaces.GPU
def macro_analysis():
    df = pd.read_csv(DB_FILE).dropna(subset=['Narrative']).reset_index(drop=True)
    if len(df) < 2:
        return None, None
    
    texts = df['Narrative'].tolist()
    # Заменяем имена на анонимные номера отчетов
    report_ids = [f"Report #{i+1}" for i in range(len(df))]
    
    # Векторизация целых документов для общей картины
    embeddings = model.encode(texts)
    sim_matrix = cosine_similarity(embeddings)
    
    # Построение Heatmap
    fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
    sns.heatmap(sim_matrix, xticklabels=report_ids, yticklabels=report_ids, 
                annot=True, cmap="YlOrRd", fmt=".2f", ax=ax_heat)
    ax_heat.set_title("Document Cosine Similarity Matrix")
    plt.tight_layout()
    
    # Построение Network Graph
    fig_graph, ax_graph = plt.subplots(figsize=(8, 6))
    G = nx.Graph()
    
    for i, r_id in enumerate(report_ids):
        G.add_node(r_id)
        
    threshold = 0.40 # Порог сходства для создания связи
    for i in range(len(report_ids)):
        for j in range(i + 1, len(report_ids)):
            if sim_matrix[i, j] > threshold:
                G.add_edge(report_ids[i], report_ids[j], weight=sim_matrix[i, j])
                
    pos = nx.spring_layout(G, seed=42)

    # Рисуем только вершины
    nx.draw_networkx_nodes(
        G, pos, node_color="skyblue", node_size=2000, ax=ax_graph
    )

    # Подписи
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_weight="bold", ax=ax_graph
    )
    
    # Рисуем толщину линий в зависимости от силы сходства
    edges = G.edges()
    weights = [G[u][v]['weight'] * 5 for u,v in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='red', alpha=0.5, ax=ax_graph)
    ax_graph.set_title(f"Semantic Network Graph (Threshold > {threshold})")

    return fig_heat, fig_graph

# ==========================================
# 6. ФУНКЦИИ МИКРО-АНАЛИЗА (Tab 3: Sentence t-SNE)
# ==========================================
@spaces.GPU
def micro_analysis():
    df = pd.read_csv(DB_FILE).dropna(subset=['Narrative']).reset_index(drop=True)
    if len(df) < 2:
        return None, "Not enough data."
    
    all_sentences = []
    parent_report_ids = []
    
    for idx, row in df.iterrows():
        sents = split_into_sentences(row['Narrative'])
        all_sentences.extend(sents)
        # Привязываем предложение к анонимному номеру отчета
        parent_report_ids.extend([f"Report #{idx+1}"] * len(sents))
        
    if len(all_sentences) < 5:
        return None, "Not enough sentences extracted."
        
    sent_embeddings = model.encode(all_sentences)
    
    perplexity = min(30, len(all_sentences) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vecs_2d = tsne.fit_transform(sent_embeddings)
    
    fig_tsne, ax_tsne = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x=vecs_2d[:,0], y=vecs_2d[:,1], hue=parent_report_ids, palette="tab10", s=100, ax=ax_tsne)
    ax_tsne.set_title("Sentence-Level Semantic Projections (t-SNE)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    sim_matrix = cosine_similarity(sent_embeddings)
    mean_sims = sim_matrix.mean(axis=1)
    top_indices = mean_sims.argsort()[-5:][::-1]
    
    central_text = "### Top 5 Most Central Semantic Fragments\n\n"
    for idx in top_indices:
        central_text += f"> **{parent_report_ids[idx]}**: \"{all_sentences[idx]}\" *(Centrality Score: {mean_sims[idx]:.2f})*\n\n"
        
    return fig_tsne, central_text

# ==========================================
# 7. ГИБРИДНЫЙ ДВИЖОК ПОИСКА (Tab 4)
# ==========================================
@spaces.GPU
def hybrid_pattern_discovery(mode, custom_motifs_text):
    df = pd.read_csv(DB_FILE).dropna(subset=['Narrative']).reset_index(drop=True)
    if len(df) < 2:
        return "Not enough data. Need at least 2 reports."
        
    all_sentences = []
    parent_report_ids = []
    
    for idx, row in df.iterrows():
        sents = split_into_sentences(row['Narrative'])
        all_sentences.extend(sents)
        parent_report_ids.extend([f"Report #{idx+1}"] * len(sents))
        
    if len(all_sentences) < 5:
        return "Not enough detailed sentences."
        
    # РЕЖИМ 1: СЛЕПОЙ ПОИСК
    if mode == "Blind Extraction (Unsupervised)":
        sent_embeddings = model.encode(all_sentences)
        num_clusters = max(2, min(8, len(all_sentences) // 10))
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(sent_embeddings)
        
        results = "### 👁️ Blind AI Extraction: Emergent Signals from Noise\n"
        results += f"*Analyzed {len(all_sentences)} sentences. Extracted {num_clusters} hidden thematic clusters without human prompts.*\n\n"
        
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) < 2: continue
                
            cluster_embeddings = sent_embeddings[cluster_indices]
            centroid = kmeans.cluster_centers_[i]
            
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances([centroid], cluster_embeddings)[0]
            global_central_idx = cluster_indices[np.argmin(distances)]
            central_sentence = all_sentences[global_central_idx]
            
            authors_in_cluster = set([parent_report_ids[idx] for idx in cluster_indices])
            
            if len(authors_in_cluster) > 1:
                results += f"#### 🟢 Discovered Archetype: *«{central_sentence[:100]}...»*\n"
                results += f"**Cross-validation:** Found in {len(authors_in_cluster)} independent reports.\n"
                count = 0
                for idx in cluster_indices:
                    if idx != global_central_idx and count < 3:
                        results += f"- *{parent_report_ids[idx]}*: \"{all_sentences[idx]}\"\n"
                        count += 1
                results += "---\n"
        return results

    # РЕЖИМ 2: ЦЕЛЕВОЙ ПОИСК
    else:
        seed_motifs = [m.strip() for m in re.split(r'[,|\n]', custom_motifs_text) if m.strip()]
        if not seed_motifs:
            return "Please enter at least one motif to search for."
            
        motif_embeddings = model.encode(seed_motifs)
        results = "### 🎯 Targeted AI Search: Hypothesis Testing\n\n"
        
        for idx_motif, motif in enumerate(seed_motifs):
            motif_emb = motif_embeddings[idx_motif]
            reports_with_motif = 0
            best_matches = []
            
            for idx_row, row in df.iterrows():
                sents = split_into_sentences(row['Narrative'])
                if not sents: continue
                
                sent_embs = model.encode(sents)
                sims = cosine_similarity([motif_emb], sent_embs)[0]
                
                max_sim = np.max(sims)
                if max_sim > 0.40: 
                    reports_with_motif += 1
                    best_idx = np.argmax(sims)
                    best_matches.append(f"*Report #{idx_row+1}*: \"{sents[best_idx]}\"")
                    
            results += f"#### ✔ Target: '{motif}'\n"
            results += f"**Found in {reports_with_motif} out of {len(df)} reports.**\n"
            if best_matches:
                for match in best_matches[:4]:
                    results += f"- {match}\n"
            results += "---\n"
            
        return results

# ----------------- ИНТЕРФЕЙС GRADIO -----------------
with gr.Blocks() as app:
    gr.Markdown("# 🌌 DreamCode: Research Platform")
    
    with gr.Tabs():
        # TAB 1: Data Ingestion
        with gr.TabItem("1. Data Ingestion"):
            with gr.Row():
                with gr.Column():
                    alias = gr.Textbox(label="Participant ID (Kept private during analysis)")
                    asc_type = gr.Dropdown(choices=["Ordinary Dream", "Lucid Dream (LD)", "OBE", "NDE", "Other"], label="State")
                    emotion = gr.Radio(choices=["Positive", "Neutral", "Negative"], label="Emotional Tone")
                    intensity = gr.Slider(minimum=1, maximum=3, step=1, label="Intensity")
                    narrative = gr.Textbox(label="Narrative", lines=7)
                    submit_btn = gr.Button("Submit Experience", variant="primary")
                    
                with gr.Column():
                    status_output = gr.Textbox(label="Status")
                    data_preview = gr.Dataframe(headers=["Timestamp", "Alias", "ASC_Type", "Emotion", "Intensity", "Narrative"])
                    
            submit_btn.click(fn=process_entry, inputs=[alias, asc_type, emotion, intensity, narrative], outputs=[status_output, data_preview])

        # TAB 2: Document Level Analysis
        with gr.TabItem("2. Document Similarity"):
            gr.Markdown("Analyze macro-connections between entire reports using Cosine Similarity (Anonymized).")
            analyze_macro_btn = gr.Button("Generate Matrix & Graph", variant="primary")
            with gr.Row():
                heat_plot = gr.Plot(label="Cosine Similarity Heatmap")
                network_plot = gr.Plot(label="Semantic Network Graph")
                
            analyze_macro_btn.click(fn=macro_analysis, inputs=[], outputs=[heat_plot, network_plot])
            
        # TAB 3: Sentence Level Analysis
        with gr.TabItem("3. Scene & Sentence t-SNE"):
            gr.Markdown("AI splits narratives into individual sentences to cluster specific scenes.")
            analyze_micro_btn = gr.Button("Process Sentences", variant="primary")
            with gr.Row():
                tsne_plot = gr.Plot(label="Sentence t-SNE")
                central_text = gr.Markdown(label="Central Fragments")
                
            analyze_micro_btn.click(fn=micro_analysis, inputs=[], outputs=[tsne_plot, central_text])

        # TAB 4: AI Pattern Discovery
        with gr.TabItem("4. AI Pattern Discovery"):
            gr.Markdown("### Semantic Search Engine\nChoose between extracting unknown signals automatically or testing specific hypotheses against the database.")
            
            search_mode = gr.Radio(
                choices=["Blind Extraction (Unsupervised)", "Targeted Search (Zero-Shot)"],
                value="Blind Extraction (Unsupervised)",
                label="Select Analysis Mode"
            )
            
            motif_input = gr.Textbox(
                label="Enter custom phrases, motifs, or fragments from your own dream (separated by commas or new lines)", 
                lines=3, 
                value="A red celestial body, Huge planet in the sky, Feeling of global catastrophe",
                visible=False 
            )
            
            def toggle_input(mode):
                if mode == "Targeted Search (Zero-Shot)":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)
                    
            search_mode.change(fn=toggle_input, inputs=[search_mode], outputs=[motif_input])
            
            analyze_btn = gr.Button("Run Analysis Engine", variant="primary")
            output_md = gr.Markdown()
            
            analyze_btn.click(fn=hybrid_pattern_discovery, inputs=[search_mode, motif_input], outputs=[output_md])

        # TAB 5: Database Explorer (NEW)
        with gr.TabItem("5. Database Explorer"):
            gr.Markdown("### 📖 Full Reports Viewer\nCross-reference the **Report ID** from the analysis tabs to read the full context of the experience here.")
            refresh_btn = gr.Button("Refresh Database")
            # wrap=True позволяет тексту переноситься на новые строки, чтобы читать абзацы целиком
            db_display = gr.Dataframe(wrap=True) 
            
            refresh_btn.click(fn=view_database, inputs=[], outputs=[db_display])
            app.load(fn=view_database, inputs=[], outputs=[db_display])

app.launch(theme=gr.themes.Monochrome())