import streamlit as st
import pandas as pd
from datetime import datetime
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import io

# Seite konfigurieren
st.set_page_config(
    page_title="Fundbüro - KI Erkennung",
    page_icon="🔍",
    layout="wide"
)

# Datenbank initialisieren
def init_database():
    conn = sqlite3.connect('fundbuero.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS gefundene_gegenstaende
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  bild BLOB,
                  klasse TEXT,
                  erstellungsdatum TEXT,
                  status TEXT DEFAULT 'gefunden',
                  rückgabedatum TEXT)''')
    conn.commit()
    conn.close()

# Modell und Labels laden
@st.cache_resource
def load_ml_components():
    try:
        model = load_model("keras_Model.h5", compile=False)
        class_names = open("labels.txt", "r").read().splitlines()
        return model, class_names
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells: {e}")
        return None, None

# Bild vorverarbeiten
def preprocess_image(image):
    # Größe anpassen
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # In Array konvertieren
    image_array = np.asarray(image)
    
    # Normalisieren
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # In die richtige Form bringen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    return data

# Bild klassifizieren
def classify_image(model, class_names, image):
    data = preprocess_image(image)
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

# Bild in Datenbank speichern
def save_to_database(image, class_name):
    conn = sqlite3.connect('fundbuero.db')
    c = conn.cursor()
    
    # Bild in Bytes konvertieren
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    
    # Datum und Zeit
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("INSERT INTO gefundene_gegenstaende (bild, klasse, erstellungsdatum) VALUES (?, ?, ?)",
              (img_bytes, class_name, now))
    conn.commit()
    conn.close()

# Bilder aus Datenbank laden
def load_from_database(class_filter=None):
    conn = sqlite3.connect('fundbuero.db')
    
    if class_filter and class_filter != "Alle":
        query = "SELECT id, bild, klasse, erstellungsdatum, status FROM gefundene_gegenstaende WHERE klasse = ? ORDER BY erstellungsdatum DESC"
        df = pd.read_sql_query(query, conn, params=(class_filter,))
    else:
        query = "SELECT id, bild, klasse, erstellungsdatum, status FROM gefundene_gegenstaende ORDER BY erstellungsdatum DESC"
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    return df

# Gegenstand als zurückgegeben markieren
def mark_as_returned(item_id):
    conn = sqlite3.connect('fundbuero.db')
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("UPDATE gefundene_gegenstaende SET status = 'zurückgegeben', rückgabedatum = ? WHERE id = ?", 
              (now, item_id))
    conn.commit()
    conn.close()

# Hauptapp
def main():
    st.title("🔍 KI-gestütztes Fundbüro")
    st.markdown("---")
    
    # Datenbank initialisieren
    init_database()
    
    # Modell laden
    model, class_names = load_ml_components()
    
    if model is None or class_names is None:
        st.error("⚠️ Bitte stelle sicher, dass 'keras_Model.h5' und 'labels.txt' im gleichen Ordner wie diese App sind.")
        return
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Gehe zu:", ["📤 Gegenstand erfassen", "🔎 Nach Gegenständen suchen", "📋 Alle Gegenstände"])
    
    if page == "📤 Gegenstand erfassen":
        st.header("📤 Gefundenen Gegenstand erfassen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Bild anzeigen
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Hochgeladenes Bild", use_container_width=True)
        
        with col2:
            if uploaded_file is not None:
                if st.button("🔍 Gegenstand erkennen und speichern", type="primary"):
                    with st.spinner("Analysiere Bild..."):
                        # Bild klassifizieren
                        class_name, confidence = classify_image(model, class_names, image)
                        
                        # In Datenbank speichern
                        save_to_database(image, class_name)
                        
                        # Ergebnis anzeigen
                        st.success(f"✅ Gegenstand wurde erkannt und gespeichert!")
                        st.info(f"**Erkannte Klasse:** {class_name}")
                        st.info(f"**Konfidenz:** {confidence:.2%}")
                        
                        # Zusätzliche Info basierend auf Klasse
                        if "flasche" in class_name.lower() or "trinkflasche" in class_name.lower():
                            st.markdown("💧 **Tipp:** Dies scheint eine Trinkflasche zu sein.")
                        elif "tshirt" in class_name.lower() or "shirt" in class_name.lower():
                            st.markdown("👕 **Tipp:** Dies scheint ein T-Shirt zu sein.")
                        elif "pullover" in class_name.lower() or "sweater" in class_name.lower():
                            st.markdown("🧥 **Tipp:** Dies scheint ein Pullover zu sein.")
    
    elif page == "🔎 Nach Gegenständen suchen":
        st.header("🔎 Nach verlorenen Gegenständen suchen")
        
        # Suchfilter
        col1, col2 = st.columns([2, 1])
        
        with col1:
            suchbegriff = st.text_input("Suchbegriff eingeben (z.B. Flasche, Tshirt, Pullover):")
        
        with col2:
            alle_klassen = ["Alle"] + class_names
            klasse_filter = st.selectbox("Oder nach Klasse filtern:", alle_klassen)
        
        if st.button("Suchen", type="primary"):
            if suchbegriff:
                # Nach Suchbegriff in den Klassen suchen
                gefilterte_klassen = [k for k in class_names if suchbegriff.lower() in k.lower()]
                if gefilterte_klassen:
                    df = load_from_database(gefilterte_klassen[0])
                else:
                    df = pd.DataFrame()
            else:
                # Nach Klasse filtern
                filter_value = None if klasse_filter == "Alle" else klasse_filter
                df = load_from_database(filter_value)
            
            if not df.empty:
                st.success(f"✅ {len(df)} Gegenstand/Gegenstände gefunden!")
                
                # Ergebnisse anzeigen
                for idx, row in df.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            # Bild aus Bytes laden und anzeigen
                            img_bytes = row['bild']
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, width=150)
                        
                        with col2:
                            st.markdown(f"**ID:** {row['id']}")
                            st.markdown(f"**Klasse:** {row['klasse']}")
                            st.markdown(f"**Gefunden am:** {row['erstellungsdatum']}")
                            st.markdown(f"**Status:** {row['status']}")
                        
                        with col3:
                            if row['status'] == 'gefunden':
                                if st.button(f"Als zurückgegeben markieren", key=f"return_{row['id']}"):
                                    mark_as_returned(row['id'])
                                    st.success("✅ Als zurückgegeben markiert!")
                                    st.rerun()
                        
                        st.markdown("---")
            else:
                st.warning("😕 Keine Gegenstände gefunden.")
    
    else:  # Alle Gegenstände
        st.header("📋 Alle erfassten Gegenstände")
        
        # Alle Daten laden
        df = load_from_database()
        
        if not df.empty:
            # Statistik
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📦 Gesamt", len(df))
            
            with col2:
                gefunden = len(df[df['status'] == 'gefunden'])
                st.metric("🔍 Noch gefunden", gefunden)
            
            with col3:
                zurueck = len(df[df['status'] == 'zurückgegeben'])
                st.metric("✅ Zurückgegeben", zurueck)
            
            st.markdown("---")
            
            # Verteilung nach Klasse
            st.subheader("Verteilung nach Klasse")
            class_distribution = df['klasse'].value_counts()
            st.bar_chart(class_distribution)
            
            st.markdown("---")
            
            # Alle Einträge anzeigen
            st.subheader("Alle Einträge")
            
            for idx, row in df.iterrows():
                with st.expander(f"Eintrag #{row['id']} - {row['klasse']} ({row['status']})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        img_bytes = row['bild']
                        img = Image.open(io.BytesIO(img_bytes))
                        st.image(img, width=200)
                    
                    with col2:
                        st.markdown(f"**ID:** {row['id']}")
                        st.markdown(f"**Klasse:** {row['klasse']}")
                        st.markdown(f"**Gefunden am:** {row['erstellungsdatum']}")
                        st.markdown(f"**Status:** {row['status']}")
                        
                        if row['status'] == 'gefunden':
                            if st.button(f"Als zurückgegeben markieren", key=f"return_all_{row['id']}"):
                                mark_as_returned(row['id'])
                                st.success("✅ Als zurückgegeben markiert!")
                                st.rerun()
        else:
            st.info("📭 Noch keine Gegenstände in der Datenbank.")

if __name__ == "__main__":
    main()
