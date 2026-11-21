# app/multimodal_app.py

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# Importer les prédicteurs
from src.vision.predict import VisionPredictor
from src.audio.predict import AudioPredictor
import tensorflow as tf
import joblib
from src.text.preprocessing import clean_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="🎭 Détecteur Multi-modal",
    page_icon="🎭",
    layout="wide"
)

# CSS personnalisé - THÈME MODERNE ET IMPRESSIONNANT
st.markdown("""
<style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-attachment: fixed;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 50%, #db2777 100%);
        padding: 1.5rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        position: relative;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 600;
        position: relative;
    }
    
    /* Warning box */
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: white;
        border-radius: 12px;
        padding: 6px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px;
        padding: 0 20px;
        font-size: 1rem;
        font-weight: 700;
        background: transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Section containers */
    .section-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .section-container:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }
    
    /* Result box - SUPER IMPRESSIONNANT */
    .result-fusion {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-fusion::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
    }
    
    .result-fusion h2 {
        font-size: 1.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        position: relative;
        text-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .result-fusion h3 {
        font-size: 2.5rem;
        font-weight: 900;
        margin: 0.5rem 0;
        position: relative;
        text-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .confidence-bar {
        background: rgba(255,255,255,0.3);
        height: 30px;
        border-radius: 15px;
        overflow: hidden;
        position: relative;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: white;
        border-radius: 15px;
        transition: width 1.5s ease-out;
        box-shadow: 0 0 15px rgba(255,255,255,0.5);
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 15px;
        border: 3px dashed #667eea;
        padding: 2rem;
        background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 900;
        padding: 1rem 2rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
    }
    
    /* Charts container */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 4rem;
        color: white;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Emotion emoji */
    .emotion-emoji {
        font-size: 3rem;
        text-shadow: 0 6px 15px rgba(0,0,0,0.3);
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
</style>
""", unsafe_allow_html=True)

# Header impressionnant
st.markdown("""
<div class="main-header">
    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🧠✨</div>
    <h1 class="main-title">Détecteur Multi-modal d'Émotions</h1>
    <p class="subtitle">📸 Vision + 🎤 Audio + ✍️ Texte = 🎯 Intelligence Artificielle</p>
</div>
""", unsafe_allow_html=True)

# Warning
st.markdown("""
<div class="warning-box">
    <h3 style="margin: 0; color: #92400e; font-size: 1rem;">⚠️ AVERTISSEMENT IMPORTANT</h3>
    <p style="margin: 0.5rem 0 0 0; color: #92400e; font-size: 0.9rem;">
        Cette application détecte les émotions apparentes. Elle <strong>NE PEUT PAS</strong> diagnostiquer de problèmes de santé mentale.
    </p>
</div>
""", unsafe_allow_html=True)

# Charger les modèles
@st.cache_resource
def load_models():
    try:
        # Vision
        vision_model_path = os.path.join(config.MODEL_DIR, 'vision_model.h5')
        vision_predictor = VisionPredictor(vision_model_path) if os.path.exists(vision_model_path) else None
        
        # Audio
        audio_model_path = os.path.join(config.MODEL_DIR, 'audio_model.h5')
        audio_encoder_path = os.path.join(config.MODEL_DIR, 'audio_label_encoder.pkl')
        audio_predictor = AudioPredictor(audio_model_path, audio_encoder_path) if os.path.exists(audio_model_path) else None
        
        # Texte
        text_model_path = os.path.join(config.MODEL_DIR, 'text_model.h5')
        text_tokenizer_path = os.path.join(config.MODEL_DIR, 'text_tokenizer.pkl')
        text_encoder_path = os.path.join(config.MODEL_DIR, 'text_label_encoder.pkl')
        
        if os.path.exists(text_model_path):
            text_model = tf.keras.models.load_model(text_model_path)
            text_tokenizer = joblib.load(text_tokenizer_path)
            text_encoder = joblib.load(text_encoder_path)
        else:
            text_model, text_tokenizer, text_encoder = None, None, None
        
        return vision_predictor, audio_predictor, (text_model, text_tokenizer, text_encoder)
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return None, None, (None, None, None)

vision_pred, audio_pred, (text_model, text_tokenizer, text_encoder) = load_models()

# Vérifier les modèles disponibles
models_available = []
if vision_pred: models_available.append("Vision 📸")
if audio_pred: models_available.append("Audio 🎤")
if text_model: models_available.append("Texte ✍️")

if not models_available:
    st.error("❌ Aucun modèle n'est disponible. Entraînez d'abord les modèles.")
    st.stop()

st.markdown(f"""
<div class="success-box">
    <p style="margin: 0; color: #065f46; font-weight: 700; font-size: 0.9rem;">
        ✅ <strong>Modèles chargés avec succès :</strong> {', '.join(models_available)}
    </p>
</div>
""", unsafe_allow_html=True)

# Emojis pour les émotions
EMOTION_EMOJIS = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fear': '😨',
    'surprise': '😲',
    'disgust': '🤢',
    'neutral': '😐'
}

EMOTION_COLORS = {
    'happy': '#10B981',
    'sad': '#3B82F6',
    'angry': '#EF4444',
    'fear': '#8B5CF6',
    'surprise': '#F59E0B',
    'disgust': '#14B8A6',
    'neutral': '#6B7280'
}

# Fonctions pour créer les graphiques
def create_radar_chart(predictions_dict):
    """Crée un graphique radar pour les probabilités d'émotions"""
    emotions = list(predictions_dict.keys())
    values = [predictions_dict[e] * 100 for e in emotions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        fillcolor='rgba(139, 92, 246, 0.4)',
        line=dict(color='rgb(139, 92, 246)', width=3),
        name='Probabilités'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(size=12, color='#666')
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(size=14, color='#333', family='Inter', weight='bold')
            ),
            bgcolor='rgba(255,255,255,0.9)'
        ),
        showlegend=False,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=14)
    )
    
    return fig

def create_modality_bar_chart(results):
    """Crée un graphique en barres pour comparer les modalités"""
    modalities = []
    emotions = []
    confidences = []
    colors = []
    icons = {'vision': '📸', 'audio': '🎤', 'text': '✍️'}
    
    for modality, result in results.items():
        modalities.append(f"{icons[modality]} {modality.upper()}")
        emotions.append(result['emotion'])
        confidences.append(result['confidence'] * 100)
        emotion_key = result['emotion'].lower()
        colors.append(EMOTION_COLORS.get(emotion_key, '#6B7280'))
    
    fig = go.Figure()
    
    for i, (mod, conf, emo, color) in enumerate(zip(modalities, confidences, emotions, colors)):
        fig.add_trace(go.Bar(
            x=[mod],
            y=[conf],
            name=emo,
            marker_color=color,
            text=[f"{emo}<br>{conf:.1f}%"],
            textposition='outside',
            textfont=dict(size=14, family='Inter', weight='bold'),
            hovertemplate=f'<b>{mod}</b><br>Émotion: {emo}<br>Confiance: {conf:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text="<b>Comparaison des Modalités</b>",
            font=dict(size=20, family='Inter', color='#333'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="",
            tickfont=dict(size=14, family='Inter')
        ),
        yaxis=dict(
            title=dict(
                text="<b>Confiance (%)</b>",
                font=dict(size=16, family='Inter')
            ),
            range=[0, 100],
            gridcolor='rgba(0,0,0,0.1)'
        ),
        showlegend=False,
        height=450,
        plot_bgcolor='rgba(255,255,255,0.9)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        margin=dict(t=80, b=60, l=60, r=40)
    )
    
    return fig

def create_emotion_distribution_pie(predictions_dict):
    """Crée un graphique circulaire pour la distribution des émotions"""
    emotions = list(predictions_dict.keys())
    values = [predictions_dict[e] * 100 for e in emotions]
    colors_list = [EMOTION_COLORS.get(e.lower(), '#6B7280') for e in emotions]
    
    fig = go.Figure(data=[go.Pie(
        labels=emotions,
        values=values,
        hole=0.4,
        marker=dict(colors=colors_list, line=dict(color='white', width=3)),
        textfont=dict(size=14, family='Inter', weight='bold'),
        hovertemplate='<b>%{label}</b><br>Probabilité: %{value:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text="<b>Distribution des Émotions</b>",
            font=dict(size=20, family='Inter', weight='bold', color='#333'),
            x=0.5,
            xanchor='center'
        ),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            font=dict(size=12)
        )
    )
    
    return fig

# Onglets pour les inputs
tab1, tab2, tab3 = st.tabs(["📸 Photo", "🎤 Audio", "✍️ Texte"])

# Variables pour stocker les inputs
uploaded_image = None
uploaded_audio = None
user_text = None

with tab1:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("### 📸 ÉTAPE 1 : Photo de votre visage")
    st.markdown("Téléchargez une photo claire de votre visage")
    uploaded_image = st.file_uploader("Choisissez une photo", type=['jpg', 'jpeg', 'png'], key='img', label_visibility="collapsed")
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="✅ Photo téléchargée", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("### 🎤 ÉTAPE 2 : Audio (optionnel)")
    st.info("💡 Enregistrez une phrase exprimant votre émotion (5-10 sec)")
    uploaded_audio = st.file_uploader("Téléchargez un fichier audio", type=['wav', 'mp3'], key='audio', label_visibility="collapsed")
    
    if uploaded_audio:
        st.audio(uploaded_audio)
        st.success("✅ Audio téléchargé")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown("### ✍️ ÉTAPE 3 : Texte (optionnel)")
    st.markdown("Décrivez votre état émotionnel")
    user_text = st.text_area(
        "Votre description",
        placeholder="Exemple : Je me sens vraiment heureux aujourd'hui !",
        height=100,
        label_visibility="collapsed"
    )
    if user_text:
        st.success(f"✅ Texte saisi : {len(user_text)} caractères")
    st.markdown('</div>', unsafe_allow_html=True)

# Bouton d'analyse
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔍 ANALYSER L'ÉMOTION MAINTENANT", type="primary", use_container_width=True):
    
    # Vérifier qu'au moins une modalité est fournie
    if not uploaded_image and not uploaded_audio and not user_text:
        st.warning("⚠️ Veuillez fournir au moins une modalité (photo, audio ou texte)")
        st.stop()
    
    with st.spinner("🔄 Analyse en cours... Intelligence Artificielle au travail..."):
        
        results = {}
        predictions_list = []
        weights = []
        all_emotion_probs = {}  # Pour le radar chart
        
        # ANALYSE VISION
        if uploaded_image and vision_pred:
            try:
                temp_img = "temp_image.jpg"
                image.save(temp_img)
                emotion_v, conf_v, probs_v = vision_pred.predict(temp_img)
                os.remove(temp_img)
                
                results['vision'] = {'emotion': emotion_v, 'confidence': conf_v}
                predictions_list.append(emotion_v)
                weights.append(0.4)
                
                # Ajouter les probabilités au dictionnaire global
                for emo, prob in probs_v.items():
                    if emo not in all_emotion_probs:
                        all_emotion_probs[emo] = []
                    all_emotion_probs[emo].append(prob)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse de l'image : {e}")
        
        # ANALYSE AUDIO
        if uploaded_audio and audio_pred:
            try:
                temp_audio = "temp_audio.wav"
                with open(temp_audio, "wb") as f:
                    f.write(uploaded_audio.read())
                
                emotion_a, conf_a, probs_a = audio_pred.predict(temp_audio)
                os.remove(temp_audio)
                
                results['audio'] = {'emotion': emotion_a, 'confidence': conf_a}
                predictions_list.append(emotion_a)
                weights.append(0.3)
                
                # Ajouter les probabilités au dictionnaire global
                for emo, prob in probs_a.items():
                    if emo not in all_emotion_probs:
                        all_emotion_probs[emo] = []
                    all_emotion_probs[emo].append(prob)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse audio : {e}")
        
        # ANALYSE TEXTE
        if user_text and text_model:
            try:
                cleaned = clean_text(user_text)
                sequence = text_tokenizer.texts_to_sequences([cleaned])
                padded = pad_sequences(sequence, maxlen=50)
                
                pred = text_model.predict(padded, verbose=0)[0]
                emotion_idx = np.argmax(pred)
                emotion_t = text_encoder.classes_[emotion_idx]
                conf_t = float(pred[emotion_idx])
                
                results['text'] = {'emotion': emotion_t, 'confidence': conf_t}
                predictions_list.append(emotion_t)
                weights.append(0.3)
                
                # Stocker toutes les probabilités pour le radar
                for i, emo in enumerate(text_encoder.classes_):
                    if emo not in all_emotion_probs:
                        all_emotion_probs[emo] = []
                    all_emotion_probs[emo].append(float(pred[i]))
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse du texte : {e}")
        
        # Moyenner les probabilités si plusieurs modalités
        if all_emotion_probs:
            all_emotion_probs = {
                emo: np.mean(probs) 
                for emo, probs in all_emotion_probs.items()
            }
        
        # FUSION DES RÉSULTATS
        if results:
            # Émotion majoritaire
            emotion_counts = Counter(predictions_list)
            final_emotion = emotion_counts.most_common(1)[0][0]
            
            # Confiance moyenne pondérée
            total_conf = sum(results[mod]['confidence'] * w 
                           for mod, w in zip(results.keys(), weights))
            final_confidence = total_conf / sum(weights) if weights else 0
            
            # AFFICHAGE DES RÉSULTATS - SUPER IMPRESSIONNANT
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Récupérer l'emoji de l'émotion
            emotion_key = final_emotion.lower()
            emoji = EMOTION_EMOJIS.get(emotion_key, '🎭')
            
            # Résultat fusionné
            st.markdown(f"""
            <div class="result-fusion fade-in">
                <div class="emotion-emoji">{emoji}</div>
                <h2>🎯 RÉSULTAT FUSIONNÉ</h2>
                <h3>{final_emotion.upper()}</h3>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {final_confidence*100}%;"></div>
                </div>
                <p style="font-size: 1.5rem; font-weight: 900; margin: 0; text-shadow: 0 4px 10px rgba(0,0,0,0.2);">
                    Confiance : {final_confidence*100:.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Vérifier cohérence
            if len(set(predictions_list)) > 1:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                     border-left: 4px solid #f59e0b; padding: 1rem; border-radius: 12px; 
                     margin: 1rem 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                    <p style="margin: 0; color: #92400e; font-weight: 700; font-size: 0.9rem;">
                        ⚠️ <strong>Émotions mixtes détectées</strong> entre les modalités.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # GRAPHIQUES - Section 1 : Comparaison des modalités
            st.markdown('<div class="section-container fade-in">', unsafe_allow_html=True)
            st.markdown("## 📊 Analyse par Modalité")
            st.markdown("Comparaison des résultats de chaque modèle :")
            
            fig_bars = create_modality_bar_chart(results)
            st.plotly_chart(fig_bars, use_container_width=True)
            
            # Détails textuels par modalité
            cols = st.columns(len(results))
            icons_modal = {'vision': '📸', 'audio': '🎤', 'text': '✍️'}
            
            for idx, (modality, result) in enumerate(results.items()):
                with cols[idx]:
                    emotion_key = result['emotion'].lower()
                    color = EMOTION_COLORS.get(emotion_key, '#6B7280')
                    emoji = EMOTION_EMOJIS.get(emotion_key, '🎭')
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {color}22 0%, {color}44 100%); 
                         padding: 1rem; border-radius: 12px; text-align: center; 
                         border: 2px solid {color};">
                        <div style="font-size: 2rem; margin-bottom: 0.3rem;">{icons_modal[modality]}</div>
                        <h4 style="margin: 0.3rem 0; font-size: 0.9rem; color: #333;">{modality.upper()}</h4>
                        <div style="font-size: 1.5rem; margin: 0.3rem 0;">{emoji}</div>
                        <p style="font-size: 1.1rem; font-weight: 700; margin: 0.3rem 0; color: {color};">
                            {result['emotion']}
                        </p>
                        <p style="font-size: 0.95rem; font-weight: 600; margin: 0; color: #666;">
                            {result['confidence']*100:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # GRAPHIQUES - Section 2 : Distribution radar (si on a les probas complètes)
            if all_emotion_probs:
                st.markdown('<div class="section-container fade-in">', unsafe_allow_html=True)
                st.markdown("## 🎯 Distribution des Émotions")
                st.markdown("Probabilités de toutes les émotions détectées :")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_radar = create_radar_chart(all_emotion_probs)
                    st.plotly_chart(fig_radar, use_container_width=True)
                
                with col2:
                    fig_pie = create_emotion_distribution_pie(all_emotion_probs)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Recommandations basées sur l'émotion
            st.markdown('<div class="section-container fade-in">', unsafe_allow_html=True)
            st.markdown("## 💡 À propos de l'Émotion")
            
            emotion_info = {
                'happy': {
                    'desc': "La joie est une émotion positive caractérisée par un sentiment de bien-être et de satisfaction.",
                    'icon': '😊',
                    'color': '#10B981'
                },
                'sad': {
                    'desc': "La tristesse est une émotion naturelle qui peut indiquer un besoin de repos ou de soutien social.",
                    'icon': '😢',
                    'color': '#3B82F6'
                },
                'angry': {
                    'desc': "La colère est une émotion puissante qui peut signaler une frustration ou une injustice perçue.",
                    'icon': '😠',
                    'color': '#EF4444'
                },
                'fear': {
                    'desc': "La peur est une émotion de protection qui nous aide à identifier et réagir aux dangers.",
                    'icon': '😨',
                    'color': '#8B5CF6'
                },
                'surprise': {
                    'desc': "La surprise est une émotion brève déclenchée par un événement inattendu.",
                    'icon': '😲',
                    'color': '#F59E0B'
                },
                'disgust': {
                    'desc': "Le dégoût est une émotion de protection qui nous éloigne de ce qui pourrait être nuisible.",
                    'icon': '🤢',
                    'color': '#14B8A6'
                },
                'neutral': {
                    'desc': "L'état neutre indique un calme émotionnel, sans émotion dominante particulière.",
                    'icon': '😐',
                    'color': '#6B7280'
                }
            }
            
            info = emotion_info.get(emotion_key, emotion_info['neutral'])
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {info['color']}22 0%, {info['color']}44 100%); 
                 padding: 1.5rem; border-radius: 15px; border-left: 4px solid {info['color']};">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                    <div style="font-size: 2.5rem;">{info['icon']}</div>
                    <h3 style="margin: 0; color: #333; font-size: 1.2rem;">{final_emotion.upper()}</h3>
                </div>
                <p style="font-size: 0.95rem; color: #444; margin: 0; line-height: 1.5;">
                    {info['desc']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Footer impressionnant
st.markdown("""
<div class="footer">
    <h3 style="margin-bottom: 1rem; font-size: 1.2rem;">🧠 Détection Multi-modale d'Émotions</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 1rem;">
        <div>
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">📊</div>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;"><strong>Limitations</strong></p>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Émotions apparentes uniquement</p>
        </div>
        <div>
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">🔬</div>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;"><strong>Technologie</strong></p>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Deep Learning multimodal</p>
        </div>
        <div>
            <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">👥</div>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;"><strong>Projet</strong></p>
            <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Module IA</p>
        </div>
    </div>
    
</div>
""", unsafe_allow_html=True)