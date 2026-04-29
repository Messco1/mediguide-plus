import streamlit as st
import torch
import torch.nn as nn
import pickle
import folium
from streamlit_folium import st_folium
from geo_api import get_coordinates, find_doctors
import re

st.set_page_config(
    page_title="MediGuide+",
    page_icon="+",
    layout="centered"
)

class MediGuideModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_specialties, num_urgency=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_specialty = nn.Linear(hidden_dim * 2, num_specialties)
        self.fc_urgency = nn.Linear(hidden_dim * 2, num_urgency)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.max(lstm_out, dim=1).values
        pooled = self.dropout(pooled)
        return self.fc_specialty(pooled), self.fc_urgency(pooled)

@st.cache_resource
def load_model():
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('data/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    model = MediGuideModel(len(vocab), 128, 256, len(le.classes_))
    model.load_state_dict(torch.load('model/best_model.pt', map_location='cpu'))
    model.eval()
    return model, vocab, le

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text, model, vocab, le):
    tokens = [vocab.get(w, 1) for w in clean_text(text).split()[:300]]
    if not tokens:
        return None, None, None
    tensor = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        spec_out, urg_out = model(tensor)
    probs = torch.softmax(spec_out, dim=1)[0]
    confidence = float(probs.max()) * 100
    specialty = le.inverse_transform([int(probs.argmax())])[0]
    urgency = int(torch.argmax(urg_out, dim=1)[0])
    return specialty, urgency, confidence

URGENCY_LABELS = {
    0: ("Low", "Regular consultation possible"),
    1: ("Normal", "Book an appointment within a few days"),
    2: ("Moderate", "Consultation recommended within 24-48h"),
    3: ("Urgent", "Seek care immediately or call 15 (SAMU)"),
}

st.title("MediGuide+")
st.caption("Intelligent medical orientation - Deep Learning + Geolocation")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=35)
with col2:
    sexe = st.selectbox("Sex", ["Male", "Female", "Other"])
with col3:
    code_postal = st.text_input("Postal code", value="75001")

symptoms = st.text_area(
    "Describe your symptoms in English",
    placeholder="Ex: I have been experiencing chest pain since this morning, with difficulty breathing and palpitations...",
    height=120
)

if "result" not in st.session_state:
    st.session_state.result = None

if st.button("Analyse symptoms", use_container_width=True):
    if not symptoms.strip():
        st.warning("Please describe your symptoms in English.")
    else:
        with st.spinner("Analysing..."):
            model, vocab, le = load_model()
            specialty, urgency, confidence = predict(symptoms, model, vocab, le)
            st.session_state.result = {
                "specialty": specialty,
                "urgency": urgency,
                "confidence": confidence,
                "code_postal": code_postal
            }

if st.session_state.result:
    r = st.session_state.result
    specialty = r["specialty"]
    urgency = r["urgency"]
    confidence = r["confidence"]
    code_postal = r["code_postal"]

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Specialty detected", specialty)
        st.progress(int(confidence), text=f"Confidence : {confidence:.1f}%")
    with col2:
        label, conseil = URGENCY_LABELS.get(urgency, URGENCY_LABELS[1])
        st.metric("Urgency level", label)
        st.caption(conseil)

    st.divider()
    st.subheader("Nearest doctors")
    lat, lon, ville = get_coordinates(code_postal)

    if lat and lon:
        st.caption(f"Location : {ville}")
        doctors = find_doctors(specialty, lat, lon)

        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker(
            [lat, lon],
            popup="Your position",
            icon=folium.Icon(color='blue', icon='home')
        ).add_to(m)
        for d in doctors:
            folium.Marker(
                [d['lat'], d['lon']],
                popup=f"{d['nom']} - {d['distance_km']} km",
                icon=folium.Icon(color='red', icon='plus-sign')
            ).add_to(m)
        st_folium(m, width=700, height=400)

        if doctors:
            for i, d in enumerate(doctors, 1):
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**{i}. {d['nom']}**")
                        st.caption(d['adresse'])
                    with c2:
                        st.markdown(f"**{d['distance_km']} km**")
        else:
            st.info("No doctor found. Try a different postal code.")
    else:
        st.error("Postal code not recognized.")

    st.divider()
    st.caption("WARNING: MediGuide+ is an orientation tool only. It does not replace professional medical advice. In case of emergency, call 15 (SAMU) or 112.")