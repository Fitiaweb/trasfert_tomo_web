https://github.com/Fitiaweb/transfert_tomo_web.git# region 1 - Importation des bibliothèques
import pydicom as dcm            # Pour lire et manipuler les fichiers médicaux DICOM
import numpy as np               # Pour les calculs mathématiques et la gestion des matrices
import pandas as pd              # Pour manipuler les tableaux de données
import os                        # Pour interagir avec le système de fichiers
import shutil                    # Pour déplacer des fichiers (archivage)
import matplotlib.pyplot as plt  # Pour tracer le graphique polaire
import time                      # Pour ajouter des petites pauses
import sqlite3                   # Pour gérer la base de données locale SQL
import json                      # Pour stocker les matrices complexes en texte
import plotly.express as px      # Pour créer des graphiques statistiques interactifs
import streamlit as st           # Bibliothèque pour créer l'interface web interactive
from fpdf import FPDF            # Pour générer le rapport PDF
# endregion

# region 2 - Page Configuration
# Configure le titre de l'onglet du navigateur et utilise toute la largeur de l'écran
st.set_page_config(page_title="Tomo Transfer", layout="wide")
# endregion

# region 3 - Automatic Folder & Database Setup
DIR_IN = r"\\nasdata1\TOMO\Transfert_tomo"  # Chemin du dossier réseau où les nouveaux DICOM sont déposés
DIR_ARCHIVE = "ARCHIVES"                    # Dossier local où les DICOM sont déplacés après traitement
DB_NAME = "tomo_database.db"                # Nom du fichier de la base de données SQLite

os.makedirs(DIR_IN, exist_ok=True)          # Crée le dossier d'entrée s'il n'existe pas
os.makedirs(DIR_ARCHIVE, exist_ok=True)     # Crée le dossier d'archive s'il n'existe pas
# endregion

# region 4 - Create SQLite database if it doesn't exist
def init_db():
    """Initialise la base de données locale et crée les tables si nécessaire."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Crée la table PATIENTS pour stocker l'ID et le nom complet
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS PATIENTS (
        Patient_ID TEXT PRIMARY KEY,
        Full_Name TEXT
    )
    ''')
    
    # Crée la table SESSIONS pour stocker chaque transfert/traitement (reliée au patient)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS SESSIONS (
        Session_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Patient_ID TEXT,
        DICOM_SOP_UID TEXT UNIQUE,
        Date_Time TEXT,
        Display_Date TEXT,
        Machine TEXT,
        Treatment_Site TEXT,
        Raw_Treatment_Site TEXT, -- Colonne de sauvegarde pour le nom brut issu du TPS
        Dose_Gy REAL,
        Nb_Frac INTEGER,
        uLCT REAL,
        Error_pct REAL,
        Profile_JSON TEXT,
        CS_mm_s REAL,    
        GP_s REAL,        
        FOREIGN KEY (Patient_ID) REFERENCES PATIENTS(Patient_ID)
    )
    ''')
    conn.commit()  # Valide les modifications
    conn.close()   # Ferme la connexion proprement

init_db() # Appelle la fonction pour s'assurer que la BDD est prête au lancement
# endregion

# region 5 - Catégorisation Dynamique
def load_categories(filepath="categories.txt"):
    """Charge le dictionnaire des localisations anatomiques depuis un fichier texte externe."""
    categories = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignore les lignes vides ou les commentaires (commençant par #)
            if not line or "=" not in line or line.startswith("#"):
                continue
            
            cat, words_str = line.split("=", 1)
            # Nettoie les mots (enlève les espaces inutiles et force la minuscule)
            words = [w.strip().lower() for w in words_str.split(",")]
            categories[cat.strip()] = words
            
    return categories

# On charge le dictionnaire une seule fois au lancement de l'application
CATEGORIES_DICT = load_categories()
# endregion

# region 6 - Données Globales (CACHE)
# @st.cache_data agit comme la "mémoire vive" (RAM) de l'application.
# Elle mémorise le résultat de cette fonction lors du premier lancement. 
# Aux clics suivants, elle redonne la donnée en mémoire au lieu de refouiller le disque dur (la base SQL),
# ce qui rend la navigation entre les onglets instantanée et fluide.
@st.cache_data
def charger_donnees_globales():
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT Patient_ID, Date_Time, Machine, Treatment_Site, Raw_Treatment_Site, Error_pct, GP_s, CS_mm_s FROM SESSIONS"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
# endregion

# region 7 - Sinogram Extraction
def get_sinogram(plan):
    """Extrait la matrice d'ouverture des lames (sinogramme) à partir du fichier RTPLAN."""
    NCP = plan.BeamSequence[0].NumberOfControlPoints  # Récupère le nombre de points de contrôle
    sinogram = np.zeros((NCP, 64))                    # Initialise une matrice vide pour le sinogramme
    cp_sequence = plan.BeamSequence[0].ControlPointSequence 

    for cp in range(NCP):                             # Boucle sur chaque point de contrôle
        try: 
            tmp = cp_sequence[cp][0x300d, 0x10a7].value # Récupère la valeur brute d'ouverture des lames
            tmp = tmp.decode('utf-8').strip('\x00').split('\\') # Décode et nettoie la chaîne de caractères
            sinogram[cp-1, :] = np.array(tmp, dtype=np.float64)   # Convertit en nombres et remplit la matrice
        except KeyError:
            continue                                         # Ignore s'il n'y a pas de données pour ce point
    return sinogram 
# endregion

# region 8 - Patient & Machine Info
def general_info(plan): 
    """Extrait l'identité du patient, le nom brut du plan et mappe le numéro de série de la machine."""
    plan_info = {} 
    plan_info["patient_id"] = str(plan.PatientID)     # Extrait l'ID du patient
    plan_info["patient_name"] = str(plan.PatientName) # Extrait le nom brut du patient
    
    # --- Recherche du nom du plan dans les balises d'identification ---
    try:
        valeurs_trouvees = []
        if (0x0030, 0x0003) in plan and plan[0x0030, 0x0003].value:
            valeurs_trouvees.append(str(plan[0x0030, 0x0003].value))
        if (0x0030, 0x0002) in plan and plan[0x0030, 0x0002].value:
            valeurs_trouvees.append(str(plan[0x0030, 0x0002].value))
        if (0x0030, 0x0004) in plan and plan[0x0030, 0x0004].value:
            valeurs_trouvees.append(str(plan[0x0030, 0x0004].value))
        if (0x300a, 0x0004) in plan and plan[0x300a, 0x0004].value:
            valeurs_trouvees.append(str(plan[0x300a, 0x0004].value))

        # Supprime les doublons de texte identiques
        valeurs_uniques = []
        for v in valeurs_trouvees:
            if v not in valeurs_uniques:
                valeurs_uniques.append(v)

        texte_brut = " | ".join(valeurs_uniques) if valeurs_uniques else "Aucun texte trouvé"
        plan_info["raw_site"] = texte_brut
        
    except Exception as e:
        plan_info["raw_site"] = "Erreur lecture"
        texte_brut = ""

    # region 9 - AUTO-CATÉGORISATION
    # Analyse le texte brut pour faire correspondre les mots-clés du fichier categories.txt
    texte_minuscule = texte_brut.lower()    # Convertit tout en minuscules pour s'affranchir de la casse
    plan_info["treatment_site"] = "Inconnu" # Valeur par défaut si aucun mot-clé ne matche
    
    for category, keywords in CATEGORIES_DICT.items():
        if any(mot in texte_minuscule for mot in keywords):
            plan_info["treatment_site"] = category
            break                           # On stoppe la recherche dès la première correspondance
    # endregion

    # region 10 - mapping
    # Association entre le numéro de série physique de l'appareil et son nom d'usage dans le service
    serial_mapping = {"4010012": "Tomo2", "210462": "Tomo4", "4010710": "Radi7"} 
    try:
        raw_serial = str(plan.DeviceSerialNumber) 
        plan_info["machine_nb"] = serial_mapping.get(raw_serial, raw_serial)
    except AttributeError:
        plan_info["machine_nb"] = "Unknown"
        
    return plan_info
# endregion

# region 11 - Traitement des fichiers DICOM
def delivery_info(plan): 
    """Extrait les paramètres cinématiques et dosimétriques de l'irradiation."""
    delivery = {}
    delivery["GP"] = float(plan.BeamSequence[0][0x300d, 0x1040].value)   # Gantry Period (Temps de rotation en s)
    delivery["PT"] = (delivery["GP"] / 51.0) * 1000.0                       # Temps par projection (ms)
    delivery["CS"] = float(plan.BeamSequence[0][0x300d, 0x1080].value)   # Vitesse de table (Couch Speed en mm/s)
    delivery["pitch"] = float(plan.BeamSequence[0][0x300d, 0x1060].value) # Facteur de pitch
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    delivery["Nrot"] = (NCP - 1) / 51                                      # Nombre de rotations totales
    delivery["TT"] = delivery["Nrot"] * delivery["GP"]                    # Temps total de traitement
    
    try:
        delivery["DS"] = float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose) # Dose par fraction
        delivery["Nb_Frac"] = int(plan.FractionGroupSequence[0].NumberOfFractionsPlanned)        # Nombre de séances
    except Exception:
        delivery["DS"] = 0.0
        delivery["Nb_Frac"] = 1
    return delivery
# endregion

# region 12 - Error Calculation
def get_error_shift(sinogram, delivery):
    """Calcule l'erreur de transfert (%), l'uLCT (%) et le profil d'erreur angulaire (ms)."""
    PT = delivery["PT"] 
    LOT_sino = PT * sinogram                   # Temps d'ouverture des lames par projection 
    maxLOT = np.max(LOT_sino)                 # Temps d'ouverture maximum
    total_lot = np.sum(LOT_sino)              # Temps d'ouverture cumulé sur tout le plan
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  # Filtre uniquement les lames qui s'ouvrent
    thresh = 18  
    undisc_LCT = 0                            # Compteur pour les erreurs LCT non discriminables
        
    error_per_projection = np.zeros(LOT_sino.shape[0]) # Stockage des erreurs par angle en MILLISECONDES
    
    # Conditions pour détecter un décalage potentiel des lames (Leaf Shift)
    cond1 = LOT_sino < (maxLOT - 1)  
    cond2 = LOT_sino > (PT - thresh) 
    row, col = np.where(cond1 & cond2) 
                                            
    for i in range(len(row)):
        if row[i] < (LOT_sino.shape[0] - 1):             
            if (LOT_sino[row[i]+1, col[i]] > (PT - 20)): 
                undisc_LCT += 1               # Incrémente si l'erreur impacte le temps de fermeture
        else: 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row > (-0.5)] 
    filtered_col = col[col > (-0.5)]
    extra_time = 0  
    
    # Calcul de la dose excessive théorique causée par l'inertie mécanique
    for i in range(len(filtered_row) - 1):
        diff = (PT - LOT_sino[filtered_row[i], filtered_col[i]]) 
        extra_time += diff  
        error_per_projection[filtered_row[i]] += diff # Cumul des temps d'erreur en ms
    
    # Retourne l'erreur transfert (%), l'uLCT (%) et le profil d'erreur complet (en ms)
    return ((extra_time / total_lot) * 100), ((undisc_LCT / (len(open_leaves_LOT))) * 100), error_per_projection
# endregion

# region 13 - File Reading
def read_files_in_directory(directory):
    """Parcourt l'arborescence du dossier cible pour lister les plans de radiothérapie (RP/RTPLAN)."""
    files_found = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(("RP", "RTPLAN")) and file.endswith(".dcm"):
                files_found.append(os.path.join(root, file))
    return files_found
# endregion

# region 14 - Générateur de Rapport PDF
def generer_rapport_pdf(patient_name, patient_id, site, budget_max, current_dose, seuil, df_history, alerte_message):
    """Génère le document PDF récapitulatif nettoyé des caractères non supportés."""
    pdf = FPDF()
    pdf.add_page()
    
    # En-tête principal
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Rapport de Transfert TomoTherapy - Controle Qualite", ln=True, align="C")
    pdf.ln(10)
    
    # Informations Patient
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Informations Patient", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 6, f"Nom : {patient_name}", ln=True)
    pdf.cell(0, 6, f"ID : {patient_id}", ln=True)
    pdf.cell(0, 6, f"Localisation : {site}", ln=True)
    pdf.ln(5)
    
    # Bilan Dosimétrique
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Bilan Dosimetrique Actuel", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 6, f"Dose Cumulee Estimee : {current_dose:.2f} Gy", ln=True)
    pdf.cell(0, 6, f"Budget Maximum Autorise (Prescription + {seuil}%) : {budget_max:.2f} Gy", ln=True)
    pdf.ln(5)
    
    # Historique des Transferts (Tableau)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Historique des Seances", ln=True)
    
    # En-tête du tableau
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(35, 8, "Date", border=1)
    pdf.cell(30, 8, "Machine", border=1)
    pdf.cell(30, 8, "Fractions", border=1)
    pdf.cell(30, 8, "Erreur (%)", border=1)
    pdf.cell(35, 8, "Dose/seance", border=1)
    pdf.ln()
    
    # Ingestion des lignes du DataFrame historique
    pdf.set_font("helvetica", "", 10)
    for index, row in df_history.iterrows():
        pdf.cell(35, 8, str(row['Date'])[:10], border=1)
        pdf.cell(30, 8, str(row['Machine']), border=1)
        pdf.cell(30, 8, str(row['Fractions Réalisées']), border=1)
        pdf.cell(30, 8, str(row['Erreur Séance (%)']), border=1)
        pdf.cell(35, 8, str(row['Dose Délivrée (Gy/s.)']), border=1)
        pdf.ln()
        
    pdf.ln(10)
    
    # Conclusion / Prévision Clinique
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Prevision de fin de traitement", ln=True)
    pdf.set_font("helvetica", "", 11)
    
    # Nettoyage des balises HTML et des Emojis pour éviter les crashs d'encodage Helvetica
    if alerte_message:
        alerte_clean = alerte_message.replace("<b>", "").replace("</b>", "").replace("<br>", "\n").replace("<i>", "").replace("</i>", "")
        alerte_clean = alerte_clean.replace("✅", "").replace("⚠️", "").replace("ℹ️", "").replace("👉", "")
        alerte_clean = alerte_clean.replace("é", "e").replace("è", "e").replace("à", "a").replace("ê", "e")
        pdf.multi_cell(0, 6, alerte_clean.strip())
    else:
        pdf.multi_cell(0, 6, "Le traitement est termine.")
    
    # Conversion du tableau d'octets mutable (bytearray) en format bytes immuable requis par Streamlit
    return bytes(pdf.output())
# endregion

# region 15 - Dashboard Engine
def display_dashboard(raw_data):
    """Génère l'affichage complet du dossier clinique du patient sélectionné."""
    if not raw_data:
        st.warning("Aucune donnée à afficher pour ce patient.")
        return

    # Extraction des constantes de session du patient
    patient_name = raw_data[0]['Patient']
    patient_id = raw_data[0]['ID']
    nb_frac_ref = raw_data[0]['Nb_Frac']
    site_traitement = raw_data[0].get('Site', 'Inconnu')  
    site_brut = raw_data[0].get('Raw_Site', 'Inconnu')    
    
    # Code HTML du bandeau supérieur d'identité
    html_header = f"""
    <div style="background-color: #f8f9fa; padding: 15px 25px; border-radius: 8px; border-left: 6px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Dossier Patient • Localisation : <span style="color: #d32f2f;">{site_traitement}</span></span>
            <h2 style="margin: 5px 0 0 0; color: #2c3e50; font-size: 26px;">{patient_name}</h2>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Séances prévues</span>
            <h3 style="margin: 5px 0 0 0; color: #2c3e50; font-size: 24px; font-weight: 700;">{nb_frac_ref}</h3>
        </div>
        <div style="text-align: center; border-left: 2px solid #e9ecef; padding-left: 20px;">
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Nom brut DICOM</span>
            <h3 style="margin: 5px 0 0 0; color: #7f8c8d; font-size: 18px;">{site_brut}</h3>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Identifiant (ID)</span>
            <h3 style="margin: 5px 0 0 0; color: #1f77b4; font-size: 22px;"># {patient_id}</h3>
        </div>
    </div>
    """
    st.markdown(html_header, unsafe_allow_html=True)

    # region 16 - Menu de correction manuelle
    with st.expander("Normaliser le nom de la localisation (Optionnel)"):
        st.markdown("<small style='color: #6c757d;'>Si le nom récupéré du DICOM comporte une faute de frappe ou est illisible, vous pouvez forcer un nom standard ici.</small>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        liste_sites_propres = list(CATEGORIES_DICT.keys()) + ["Autre", "Inconnu"]

        if site_traitement not in liste_sites_propres:
            options_affichage = [site_traitement] + liste_sites_propres
            default_idx = 0
        else:
            options_affichage = liste_sites_propres
            default_idx = liste_sites_propres.index(site_traitement)
            
        with col1:
            nouveau_site = st.selectbox("Sélectionnez la bonne localisation :", options_affichage, index=default_idx, key=f"select_site_{patient_id}")
            
        with col2:
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("Valider la correction", type="primary", use_container_width=True, key=f"btn_site_{patient_id}"):
                conn_update = sqlite3.connect(DB_NAME)
                cursor_update = conn_update.cursor()
                cursor_update.execute("""
                    UPDATE SESSIONS 
                    SET Treatment_Site = ? 
                    WHERE Patient_ID = ?
                """, (nouveau_site, patient_id))
                conn_update.commit()
                conn_update.close()
                
                st.toast("Nom mis à jour avec succès !", icon="✅")
                time.sleep(0.5) 
                st.rerun() 
                
    st.markdown("<br>", unsafe_allow_html=True)
    # endregion

    # region 17 - Saisie des fractions (Validation par le clinicien)
    st.markdown("##### Validation des séances réalisées")
    st.markdown("<small style='color: #6c757d;'>Le DICOM ne reflétant pas toujours la réalité clinique de Mosaiq/Aria, merci de valider le nombre de séances réellement effectuées.</small>", unsafe_allow_html=True)
    
    fractions_done_list = []
    cols = st.columns(len(raw_data)) # Génère dynamiquement une colonne par plan machine importé
    
    for index in range(len(raw_data)):
        machine = raw_data[index]['Machine']
        if index < len(raw_data) - 1:
            # Soustraction automatique entre les deux plans successifs pour deviner le réalisé historique
            diff_auto = int(max(0, raw_data[index]['Nb_Frac'] - raw_data[index+1]['Nb_Frac']))
            with cols[index]:
                f_done = st.number_input(f"Faites sur {machine}", min_value=0, max_value=int(nb_frac_ref), value=diff_auto, key=f"frac_{index}")
            fractions_done_list.append(f_done)
        else:
            # Configuration par défaut pour le plan actif du jour
            with cols[index]:
                f_done = st.number_input(f"Aujourd'hui sur {machine}", min_value=0, max_value=int(nb_frac_ref), value=0, key=f"frac_{index}")
            fractions_done_list.append(f_done)
    st.markdown("<br>", unsafe_allow_html=True)
    # endregion

    # region 18 - CALCUL DU BUDGET DOSE : Dose totale + % du seuil
    seuil_patient = st.session_state.get("seuil_global_val", 1.5)
    total_cumulated_dose = 0.0 
    final_table_data = [] 
    
    try:
        dose_nominal_ref = raw_data[0]['Dose (Gy)']
        dose_totale_prescrite = dose_nominal_ref * nb_frac_ref
        total_budget_Gy = dose_totale_prescrite * (1 + (seuil_patient / 100.0))
    except:
        total_budget_Gy = 0.0
        dose_totale_prescrite = 0.0

    for index, item in enumerate(raw_data):
        error_session_pct = item['Session Error (%)']
        dose_nominal = item['Dose (Gy)']
        fractions_done = fractions_done_list[index]
        
        # Intégration physique de l'erreur mécanique sur le calcul de la dose réelle par séance
        actual_dose_session = dose_nominal * (1 + (error_session_pct / 100.0)) 
        
        if index == 0:
            date_display = item['Date'] + " (Initiale)" 
            total_cumulated_dose += (dose_nominal * fractions_done)
            cumul_str = f"{total_cumulated_dose:.2f}"
            comment = f"Prescription : {dose_totale_prescrite:.2f} Gy"
            alert = False
    # endregion

    # region 19 - Préparation du tableau final dossier patient pour affichage
            final_table_data.append({
                "Date": date_display, 
                "Machine": item['Machine'],
                "Fractions Réalisées": fractions_done,
                "Dose Prévue (Gy/s.)": f"{dose_nominal:.2f}", 
                "uLCT (%)": f"{item['uLCT (%)']:.2f}",
                "Erreur Séance (%)": "-",
                "Dose Délivrée (Gy/s.)": "-", 
                "Dose Cumulée (Gy)": cumul_str, 
                "Commentaire": comment, 
                "_Alert": alert, "_Index": index, "_Cumul_Val": total_cumulated_dose, "_Budget_Total": total_budget_Gy, "_Nb_Frac_Plan": item['Nb_Frac']
            })
        else:
            total_cumulated_dose += (actual_dose_session * fractions_done)
            cumul_str = f"{total_cumulated_dose:.2f}"
    # endregion

    # region 20 - Système d'alerte : basé sur le % d'erreur
            comments = []
            alert = False
            if error_session_pct > seuil_patient: 
                comments.append(f"Erreur > {seuil_patient}%")
                alert = True
                
            comment = " | ".join(comments) if comments else "OK"
        
            final_table_data.append({
                "Date": item['Date'], 
                "Machine": item['Machine'],
                "Fractions Réalisées": fractions_done,
                "Dose Prévue (Gy/s.)": f"{dose_nominal:.2f}", 
                "uLCT (%)": f"{item['uLCT (%)']:.2f}",
                "Erreur Séance (%)": f"{error_session_pct:.2f}",
                "Dose Délivrée (Gy/s.)": f"{actual_dose_session:.2f}",
                "Dose Cumulée (Gy)": cumul_str, 
                "Commentaire": comment, 
                "_Alert": alert, "_Index": index, "_Cumul_Val": total_cumulated_dose, "_Budget_Total": total_budget_Gy, "_Nb_Frac_Plan": item['Nb_Frac']
            })

    df = pd.DataFrame(final_table_data) 
    # endregion
    
    # region 21 - Fonction pour colorer les colonnes
    def style_dataframe(row):
        styles = []
        line_style = ''
        if row['_Index'] == 0: 
            line_style = 'background-color: #f0f8ff; font-weight: bold;' 
        elif row['_Alert']: 
            line_style = 'background-color: #ffebee; color: #d32f2f; font-weight: bold;' 
            
        for col in row.index:
            cell_style = line_style 
            if col == 'Machine':
                if row['Machine'] == 'Tomo2': cell_style = 'background-color: #bbdefb; color: #000000; font-weight: bold;' 
                elif row['Machine'] == 'Tomo4': cell_style = 'background-color: #ffcdd2; color: #000000; font-weight: bold;' 
                elif row['Machine'] == 'Radi7': cell_style = 'background-color: #c8e6c9; color: #000000; font-weight: bold;' 
            styles.append(cell_style)
        return styles
    # endregion

    # region 22 - Affichage du tableau final
    styled_df = df.style.apply(style_dataframe, axis=1) 
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        height=200,
        column_config={
            "_Alert": None,      
            "_Index": None,
            "_Cumul_Val": None,
            "_Budget_Total": None,
            "_Nb_Frac_Plan": None
        }
    )
    st.markdown("---")
    # endregion

    col_graph1, col_graph2 = st.columns([1, 1]) 

    # region 23 - Bloc de gauche : PRÉVISION DE FIN DE TRAITEMENT
    with col_graph1:
        if len(df) > 0:
            last_session_df = df.iloc[-1]
            budget_max =  last_session_df['_Budget_Total']
            current_dose = last_session_df['_Cumul_Val']
            st.subheader(" Prévision de fin de traitement")
            alerte_message = ""

            if len(df) > 1:
                machine_actuelle = last_session_df['Machine']
                
                if last_session_df['Dose Délivrée (Gy/s.)'] == "-":
                    dose_par_seance_actuelle = float(last_session_df['Dose Prévue (Gy/s.)'])
                else:
                    dose_par_seance_actuelle = float(last_session_df['Dose Délivrée (Gy/s.)'])
                
                total_done_simul = sum(fractions_done_list)
                theoretical_remaining_sessions = nb_frac_ref - total_done_simul
                
                budget_remaining = budget_max - current_dose
                max_sessions_possibles = int(budget_remaining / dose_par_seance_actuelle) if dose_par_seance_actuelle > 0 and budget_remaining > 0 else 0
                
                dose_finale_projetee = current_dose + (theoretical_remaining_sessions * dose_par_seance_actuelle)
                
                if theoretical_remaining_sessions > 0:
                    if max_sessions_possibles == theoretical_remaining_sessions:
                        alerte_bg = "#d4edda" 
                        alerte_text = "#155724"
                        alerte_message = f" <b>Rythme Conforme :</b> Le patient peut faire ses <b>{theoretical_remaining_sessions} séances restantes</b> sur la {machine_actuelle} sans dépasser la limite de dose."
                    elif max_sessions_possibles < theoretical_remaining_sessions:
                        alerte_bg = "#f8d7da" 
                        alerte_text = "#721c24"
                        perte = theoretical_remaining_sessions - max_sessions_possibles
                        alerte_message = f" <b>ALERTE SURDOSE :</b> Au rythme de la {machine_actuelle} ({dose_par_seance_actuelle:.2f} Gy/s.), faire les {theoretical_remaining_sessions} séances prévues fera dépasser la limite autorisée ({dose_finale_projetee:.2f} Gy).<br><br>👉 Il ne peut faire que <b>{max_sessions_possibles} séances supplémentaires maximum</b> (soit <b>-{perte} séance(s)</b> à annuler sur l'ordonnance)."
                    else:
                        alerte_bg = "#fff3cd" 
                        alerte_text = "#856404"
                        gain = max_sessions_possibles - theoretical_remaining_sessions
                        alerte_message = f" <b>SOUS-DOSAGE :</b> La {machine_actuelle} délivrant moins que prévu, la limite ne sera pas atteinte à la fin du traitement. Il y a une marge pour rajouter <b>+{gain} séance(s)</b> si le médecin le juge nécessaire."

                    html_alerte = f"""
                    <div style="background-color: {alerte_bg}; padding: 16px; border-radius: 8px; color: {alerte_text}; margin-top: 10px; border: 1px solid {alerte_text};">
                        {alerte_message}
                    </div>
                    """
                    st.markdown(html_alerte, unsafe_allow_html=True)
                else:
                    st.success("Le traitement est théoriquement terminé (Toutes les séances ont été réalisées).")
            else:
                dose_last = float(last_session_df['Dose Prévue (Gy/s.)'])
                st.info(f"Le patient n'a subi aucun transfert. Il reste {nb_frac_ref - sum(fractions_done_list)} séances prévues sur la machine d'origine.")

            st.markdown("<br><hr><br>", unsafe_allow_html=True)
    # endregion

    # region 24 - Bloc de gauche : BILAN DOSIMÉTRIQUE CUMULÉ
            st.subheader(" Bilan Dosimétrique Cumulé") 

            # Calcul décimal du pourcentage pour la jauge progress (1.0 = 100%)
            percentage = min(current_dose / budget_max, 1.0) if budget_max > 0 else 0.0
            
            bg_color = "#e8f5e9" if percentage < 1.0 else "#ffebee"
            text_color = "#2e7d32" if percentage < 1.0 else "#c62828"
            
            html_budget = f"""
            <div style='background-color: {bg_color}; padding: 20px; border-radius: 10px; border: 2px solid {text_color}; text-align: center; margin-bottom: 10px;'>
                <h4 style='color: {text_color}; margin: 0; font-weight: 600;'>Dose Cumulée Estimée</h4>
                <h1 style='color: {text_color}; margin: 5px 0; font-size: 38px;'>{current_dose:.2f} <span style='font-size: 20px; font-weight: normal;'>/ {budget_max:.2f} Gy</span></h1>
                <p style='margin: 0; color: #555; font-size: 14px;'><i>Limite maximale tolérée : Prescription + {seuil_patient}%</i></p>
            </div>
            """
            st.markdown(html_budget, unsafe_allow_html=True)
            st.progress(percentage) 
    # endregion

    # region 25 - Bloc de gauche : BOUTON PDF (Export)
            st.markdown("<br>", unsafe_allow_html=True)
            pdf_bytes = generer_rapport_pdf(
                patient_name=patient_name,
                patient_id=patient_id,
                site=site_traitement,
                budget_max=budget_max,
                current_dose=current_dose,
                seuil=seuil_patient,
                df_history=df,
                alerte_message=alerte_message
            )
            
            st.download_button(
                label="📄 Exporter le Rapport Clinique (PDF)",
                data=pdf_bytes,
                file_name=f"Rapport_Tomo_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    # endregion

    # region 26 - Bloc de droite : Graphique Polaire de l'Erreur
    with col_graph2:
        st.subheader("Localisation angulaire") 
        
        transfer_sessions = []
        if len(raw_data) > 0:
            previous_machine = raw_data[0]['Machine']
            for item in raw_data[1:]: 
                current_machine = item['Machine']
                if current_machine != previous_machine:
                    transfer_sessions.append(item)
                    previous_machine = current_machine
        
        if len(transfer_sessions) == 0:
            st.info("No machine transfer detected for this patient. The treatment remained on the initial machine.")
        else:
            menu_options = []
            for s in transfer_sessions:
                menu_options.append(f"Transfer: {s['Date']} to {s['Machine']}")
                    
            selected_date = st.selectbox("Select the transfer event to analyze:", options=menu_options)
            chosen_index = menu_options.index(selected_date)
            session_to_analyze = transfer_sessions[chosen_index]

            if session_to_analyze['Session Error (%)'] > 0.0:
                total_errors = np.array(session_to_analyze['Profile_Error']) 
                
                CS = session_to_analyze.get('CS', 0) 
                GP = session_to_analyze.get('GP', 0) 
                
                distance_tour_cm = (CS * GP) / 10.0 if CS and GP else 0  
                n_rotations = len(total_errors) // 51                    
                
                if n_rotations > 1 and distance_tour_cm > 0:
                    options_cm = []
                    for i in range(1, n_rotations + 1):
                        raw_pos = i * distance_tour_cm
                        rounded_pos = round(raw_pos, 1)
                        text = f"{rounded_pos} cm"
                        options_cm.append(text)

                    selection = st.select_slider("Table advancement on the longitudinal axis (Gz)", options=options_cm) 
                    rotation_target = options_cm.index(selection) + 1 
                    
                    start = (rotation_target - 1) * 51 
                    end = start + 51
                    slice_errors = total_errors[start:end] 
                    caption_text = f"Temps d'erreur sur la rotation {rotation_target} (Gantry 0° - 360°)."
                else:
                    slice_errors = total_errors
                    caption_text = "Temps d'erreur sur 1 rotation (Gantry 0° - 360°)."
                
                N_total = len(slice_errors)
                angles = np.linspace(0, 2 * np.pi, N_total, endpoint=False) 
                
                angles_closed = np.concatenate((angles, [angles[0]])) 
                errors_closed = np.concatenate((slice_errors, [slice_errors[0]])) 
                
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5, 5.5)) 
                ax.set_theta_zero_location("N") 
                ax.set_theta_direction(-1)      
       
                max_err_global = np.max(total_errors) 
                if max_err_global == 0:
                    max_err_global = 0.1
                ax.set_ylim(max_err_global * 1.3, 0) 

                ax.fill_between(angles_closed, 0, errors_closed, color='#ff4757', alpha=0.35) 
                ax.plot(angles_closed, errors_closed, color='#c0392b', linewidth=2.0, zorder=3)
                
                # --- MODIFICATION ICI : Grille sur 51 projections (Toutes affichées) ---
                angles_proj = np.linspace(0, 2 * np.pi, 51, endpoint=False)
                ax.set_xticks(angles_proj) 
                
                # Création des labels de 1 à 51
                labels_proj = [str(i + 1) for i in range(51)]
                        
                ax.set_xticklabels(labels_proj, fontsize=6, color='#2c3e50') 
                # -----------------------------------------------------------------------

                ax.set_facecolor('white') 
                tick_values = [max_err_global * 0.25, max_err_global * 0.5, max_err_global * 0.75, max_err_global]
                ax.set_yticks(tick_values) 
                
                labels_ticks = [f"{val:.1f} ms" for val in tick_values]
                ax.set_yticklabels(labels_ticks, fontsize=7, color='#d32f2f', fontweight='bold') 
                
                ax.set_rlabel_position(25)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False) 
                st.caption(caption_text)
            else:
                st.info("No error detected in the selected session.")
# endregion

# region 27 - Barre latérale accueil
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Département de Physique Médicale**")

# endregion

# region 28 - BOUTON PROCÉDURE (Téléchargement)
doc_path = os.path.join(os.path.dirname(__file__), "Procédure2.docx")
try:
    with open(doc_path, "rb") as file:
        st.sidebar.download_button(
            label=" Télécharger la Procédure",
            data=file,
            file_name="Procédure_Tomo_Transfer.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
except FileNotFoundError:
    st.sidebar.warning(" Fichier de procédure introuvable sur le réseau.")
# endregion

# region 29 - Récupération auto des fichiers DICOM et données dans le dossier d'entrée
files_in = read_files_in_directory(DIR_IN)

if len(files_in) > 0:
    with st.sidebar.status(f"Intégration automatique de {len(files_in)} nouveau(x) plan(s)...", expanded=True) as status:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        duplicates_ignored = 0
        new_processed = 0

        for f in files_in:
            try:
                plan = dcm.dcmread(f) 
                try:
                    uid = plan.SOPInstanceUID
                except AttributeError:
                    continue 
                
                try:
                    info = general_info(plan)
                    delivery = delivery_info(plan)
                    sinogram = get_sinogram(plan)
                    data = get_error_shift(sinogram, delivery) 
                    
                    parts = info["patient_name"].split("^")  
                    full_name = f"{parts[1] if len(parts) > 1 else ''} {parts[0]}".strip()
                    
                    try: plan_date = str(plan[0x0008, 0x0012].value) 
                    except KeyError: plan_date = "00000000" 
                    try: plan_time = str(plan[0x0008, 0x0013].value) 
                    except KeyError: plan_time = "000000"
                    
                    date_time_sort = f"{plan_date}{plan_time}"
                    display_date = f"{plan_date[6:8]}/{plan_date[4:6]}/{plan_date[0:4]} à {plan_time[0:2]}:{plan_time[2:4]}:{plan_time[4:6]}"
                    
                    cursor.execute("INSERT OR IGNORE INTO PATIENTS (Patient_ID, Full_Name) VALUES (?, ?)", 
                                   (info["patient_id"], full_name))
                                   
                    profile_json = json.dumps(data[2].tolist())
                    
                    cursor.execute("""
                        INSERT INTO SESSIONS (
                            Patient_ID, DICOM_SOP_UID, Date_Time, Display_Date, Machine, Treatment_Site, Raw_Treatment_Site, 
                            Dose_Gy, Nb_Frac, uLCT, Error_pct, Profile_JSON, CS_mm_s, GP_s
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (info["patient_id"], uid, date_time_sort, display_date, info["machine_nb"], 
                          info["treatment_site"], info["raw_site"],
                          delivery["DS"], delivery.get("Nb_Frac", 1), data[1], data[0], profile_json,
                          delivery["CS"], delivery["GP"]))
                    
                    new_processed += 1
                    
                except sqlite3.IntegrityError:
                    duplicates_ignored += 1 
                    
            except Exception as e:
                st.sidebar.error(f"Erreur sur {os.path.basename(f)} : {e}")
            
            finally:
                file_name = os.path.basename(f)
                dest_path = os.path.join(DIR_ARCHIVE, file_name)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(f, DIR_ARCHIVE)

                parent_dir = os.path.dirname(f)
                if parent_dir != DIR_IN and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
        
        conn.commit()
        conn.close()
        status.update(label="Traitement terminé !", state="complete", expanded=False)

    if new_processed > 0:
        charger_donnees_globales.clear()
        st.toast(f"{new_processed} plan(s) traité(s) avec succès !", icon="✅")
        time.sleep(1)
        st.rerun() 
else:
    st.sidebar.success(" Base de données à jour. En attente de nouveaux DICOM...")
# endregion

# region 30 - GESTION DE LA NAVIGATION (passer de l'accueil au dossier patient)
if "vue_actuelle" not in st.session_state:
    st.session_state.vue_actuelle = "Accueil" 
if "patient_cible" not in st.session_state:
    st.session_state.patient_cible = None
# endregion

# region 31 - MENU DE NAVIGATION Accueil & Statistiques
if st.session_state.vue_actuelle == "Accueil":
    st.markdown("###  Contrôle Qualité Global du Service")
    
    seuil_alerte = st.number_input(
        "Définir le seuil d'alerte clinique par transfert (%) :", 
        min_value=0.0, max_value=10.0, value=1.5, step=0.1, key="seuil_global",
        help="Seuil de tolérance pour une séance individuelle."
    )
    st.session_state["seuil_global_val"] = seuil_alerte
    st.markdown("<br>", unsafe_allow_html=True)

    df_stats = charger_donnees_globales()

    if len(df_stats) == 0:
        st.info("Aucune donnée disponible pour le moment. Ingérez des fichiers DICOM pour générer les statistiques.")
    else:
        df_stats = df_stats.sort_values(by=['Patient_ID', 'Date_Time'])
        
        conn_all = sqlite3.connect(DB_NAME)
        cursor_all = conn_all.cursor()
        cursor_all.execute("""
            SELECT s.Patient_ID, p.Full_Name, s.Machine, s.Dose_Gy, s.Nb_Frac, s.Error_pct, s.Treatment_Site, s.Display_Date, s.Date_Time
            FROM SESSIONS s
            JOIN PATIENTS p ON s.Patient_ID = p.Patient_ID
            ORDER BY s.Date_Time ASC
        """)
        toutes_sessions = cursor_all.fetchall()
        conn_all.close()

        if len(toutes_sessions) > 0:
            patients_dict = {}
            for row in toutes_sessions:
                pid = row[0]
                if pid not in patients_dict:
                    patients_dict[pid] = []
                patients_dict[pid].append(row)
            
            tableau_final = []
            
            for pid, sessions in patients_dict.items():
                plan_initial = sessions[0]
                plan_actuel = sessions[-1] 
                
                nom_patient = plan_actuel[1]
                machine_actuelle = plan_actuel[2]
                dose_actuelle = plan_actuel[3]
                frac_restantes = plan_actuel[4]
                erreur_actuelle = plan_actuel[5]
                site = plan_actuel[6]
                date_display = plan_actuel[7]
                date_sort = plan_actuel[8] 
                
                nb_frac_ref = plan_initial[4] # Nombre total de séances prescrites initialement
                
                if len(sessions) > 1: 
                    dose_init = plan_initial[3]
                    frac_init = plan_initial[4]
                    dose_totale_prescrite = dose_init * frac_init
                    budget_theorique = dose_totale_prescrite * (1 + (seuil_alerte / 100.0))
                    
                    alerte_erreur = erreur_actuelle > seuil_alerte
                    
                    if alerte_erreur:
                        statut = "🔴 % Trop haut"
                        
                        # Calcul dynamique en amont du nombre de séances maximales autorisées
                        current_dose_sim = 0.0
                        fractions_accumulated_sim = 0
                        for i in range(len(sessions) - 1):
                            idx_dose = sessions[i][3]
                            idx_err = sessions[i][5]
                            idx_frac_planned = sessions[i][4]
                            next_frac_planned = sessions[i+1][4]
                            diff_auto = int(max(0, idx_frac_planned - next_frac_planned))
                            
                            actual_dose_per_frac = idx_dose * (1 + (idx_err / 100.0)) if i > 0 else idx_dose
                            current_dose_sim += actual_dose_per_frac * diff_auto
                            fractions_accumulated_sim += diff_auto
                        
                        # Calcul de l'impact sur la dernière machine
                        last_dose = plan_actuel[3]
                        last_err = plan_actuel[5]
                        actual_last_dose_per_frac = last_dose * (1 + (last_err / 100.0))
                        
                        budget_remaining_sim = budget_theorique - current_dose_sim
                        max_sessions_possibles_sim = int(budget_remaining_sim / actual_last_dose_per_frac) if actual_last_dose_per_frac > 0 and budget_remaining_sim > 0 else 0
                        
                        seances_max_calc = fractions_accumulated_sim + max_sessions_possibles_sim
                        seances_max_affichage = min(seances_max_calc, nb_frac_ref)
                    else:
                        statut = "🟢 Conforme"
                        seances_max_affichage = nb_frac_ref
                    
                    erreur_str = f"{erreur_actuelle:.2f} %"
                else: 
                    statut = "⚪ Plan Initial"
                    erreur_str = "-"
                    seances_max_affichage = nb_frac_ref
                
                tableau_final.append({
                    "Date_Sort": date_sort, 
                    "Date (Dernier import)": date_display,
                    "ID Patient": pid,
                    "Nom": nom_patient,
                    "Localisation": site,
                    "Machine Destination": machine_actuelle,
                    "Erreur Transfert": erreur_str,
                    "Séances Max Réalisables": f"{seances_max_affichage} / {nb_frac_ref}",
                    "Statut": statut
                })
            
            df_recap = pd.DataFrame(tableau_final)
            df_recap = df_recap.sort_values(by="Date_Sort", ascending=False).drop(columns=["Date_Sort"])
            
            def colorer_statut(row):
                return [
                    'color: #c62828; font-weight: bold' if col == 'Statut' and "🔴" in row['Statut'] 
                    else 'color: #9e9e9e; font-style: italic' if "⚪" in row['Statut'] 
                    else '' 
                    for col in row.index
                ]
            
            st.markdown(" **Cliquez directement sur la ligne d'un patient pour ouvrir son dossier clinique complet.**")
            st.info(" **Navigation :** Cochez la petite case située tout à gauche de la ligne d'un patient pour ouvrir son dossier détaillé.")
            
            event = st.dataframe(
                df_recap.style.apply(colorer_statut, axis=1),
                use_container_width=True,
                hide_index=True,
                height=400,
                on_select="rerun",           
                selection_mode="single-row"  
            )
            
            if len(event.selection.rows) > 0:
                index_clique = event.selection.rows[0]
                patient_id_clique = df_recap.iloc[index_clique]["ID Patient"]
                
                st.session_state.patient_cible = patient_id_clique
                st.session_state.vue_actuelle = "Dossier"
                st.rerun() 
        
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### Statistiques Globales")
        
        sub_tab_loc, sub_tab_mach, sub_tab_time, sub_tab_complex = st.tabs([
            " Par Localisation", 
            " Par Sens de Transfert", 
            " Évolution dans le Temps",
            " Complexité vs Erreur"
        ])
# endregion

        # region 32 - SOUS-ONGLET A : Localisation
        with sub_tab_loc:
            st.markdown("<small style='color: #6c757d;'>Erreur moyenne des transferts par zone traitée (Plans initiaux exclus).</small><br>", unsafe_allow_html=True)
            
            df_loc = df_stats.dropna(subset=['Treatment_Site']).copy()
            df_loc['session_num'] = df_loc.groupby('Patient_ID').cumcount() 
            df_transfers_loc = df_loc[df_loc['session_num'] > 0]            
            
            if len(df_transfers_loc) == 0:
                st.warning("Il n'y a pas encore eu de transfert de machine enregistré dans la base.")
            else:
                df_mean_loc = df_transfers_loc.groupby('Treatment_Site')['Error_pct'].mean().reset_index()
                df_mean_loc = df_mean_loc.sort_values(by='Error_pct', ascending=False)

                fig_loc = px.bar(
                    df_mean_loc,
                    x='Treatment_Site',
                    y='Error_pct',
                    title="Erreur Moyenne par catégorie de localisation",
                    labels={'Treatment_Site': 'Localisation Anatomique', 'Error_pct': 'Erreur Moyenne (%)'},
                    text_auto='.2f', 
                    color='Error_pct', 
                    color_continuous_scale='Reds' 
                )

                fig_loc.update_layout(xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
                fig_loc.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text=f"Seuil ({seuil_alerte}%)")
                st.plotly_chart(fig_loc, use_container_width=True)
        # endregion

        # region 33 - SOUS-ONGLET B : Sens de Transfert
        with sub_tab_mach:
            st.markdown("<small style='color: #6c757d;'>Analyse par trajet de transfert et taux de dépassement du seuil clinique.</small><br>", unsafe_allow_html=True)

            df_mach = df_stats.copy()
            df_mach['Prev_Machine'] = df_mach.groupby('Patient_ID')['Machine'].shift(1)
            df_transitions = df_mach.dropna(subset=['Prev_Machine']).copy()
            df_transitions = df_transitions[df_transitions['Machine'] != df_transitions['Prev_Machine']]

            if len(df_transitions) == 0:
                st.info("Aucun changement inter-machines détecté dans la base pour le moment.")
            else:
                df_transitions['Trajet'] = df_transitions['Prev_Machine'] + " ➔ " + df_transitions['Machine']
                df_transitions['Depasse_Seuil'] = df_transitions['Error_pct'] > seuil_alerte 
                
                liste_trajets = sorted(df_transitions['Trajet'].unique())
                trajet_choisi = st.selectbox(" Filtrer les compteurs et la répartition pour un trajet grandiose :", ["Tous les transferts (Global)"] + liste_trajets)
                if trajet_choisi == "Tous les transferts (Global)": 
                    df_focus = df_transitions
                    titre_pie = "Répartition Globale"
                else:
                    df_focus = df_transitions[df_transitions['Trajet'] == trajet_choisi]
                    titre_pie = f"Répartition : {trajet_choisi}"

                total_focus = len(df_focus)
                depassements_focus = df_focus['Depasse_Seuil'].sum()
                taux_focus = (depassements_focus / total_focus * 100) if total_focus > 0 else 0
                
                col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                col_kpi1.metric(f"Total de Transferts", total_focus)
                col_kpi2.metric(f"Transferts > {seuil_alerte}% (Alerte)", int(depassements_focus))
                col_kpi3.metric(f"Taux d'alerte", f"{taux_focus:.1f} %")
                
                st.markdown("---")

                col_chart1, col_chart2 = st.columns([2, 1])
                
                df_traj_stats = df_transitions.groupby('Trajet').agg(
                    Error_mean=('Error_pct', 'mean'),
                    Depassement_rate=('Depasse_Seuil', lambda x: x.mean() * 100),
                    Count=('Error_pct', 'count')
                ).reset_index()
                
                df_traj_stats = df_traj_stats.sort_values(by='Error_mean', ascending=False)
                
                with col_chart1:
                    fig_mach = px.bar(
                        df_traj_stats,
                        x='Trajet',
                        y='Error_mean',
                        title="Erreur Moyenne par Trajet (Vue Globale Comparative)",
                        labels={
                            'Trajet': 'Sens du transfert', 
                            'Error_mean': 'Erreur Moyenne (%)',
                            'Depassement_rate': f"Taux d'alerte (>{seuil_alerte}%)",
                            'Count': 'Nombre de patients'
                        },
                        text_auto='.2f', 
                        color='Error_mean', 
                        color_continuous_scale='Oranges',
                        hover_data={'Depassement_rate': ':.1f', 'Count': True}
                    )
                    
                    fig_mach.update_traces(hovertemplate='<b>%{x}</b><br>Erreur Moyenne: %{y:.2f}%<br>Patients en alerte: %{customdata[0]:.1f}%<br>Nb total de patients: %{customdata[1]}<extra></extra>')
                    fig_mach.update_layout(xaxis_tickangle=0, plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0, l=0, r=0))
                    fig_mach.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text=f"Seuil ({seuil_alerte}%)")
                    st.plotly_chart(fig_mach, use_container_width=True)

                with col_chart2:
                    if total_focus > 0:
                        fig_pie = px.pie(
                            names=[f'< {seuil_alerte}% (Conformes)', f'> {seuil_alerte}% (Alertes)'],
                            values=[total_focus - depassements_focus, depassements_focus],
                            title=titre_pie,
                            color_discrete_sequence=['#2ecc71', '#e74c3c'], 
                            hole=0.4 
                        )
                        fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), showlegend=False)
                        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info("Aucune donnée pour tracer le graphique.")
        # endregion

        # region 34 - SOUS-ONGLET C : Évolution dans le Temps
        with sub_tab_time:
            st.markdown("<small style='color: #6c757d;'>Suivi temporel de l'erreur moyenne pour détecter la fatigue des machines (maintenance prédictive). <b>Les plans initiaux sont exclus.</b></small><br><br>", unsafe_allow_html=True)
            
            df_time = df_stats.copy()

            df_time = df_time.sort_values(by=['Patient_ID', 'Date_Time'])
            df_time['session_num'] = df_time.groupby('Patient_ID').cumcount()
            df_time = df_time[df_time['session_num'] > 0] 
            
            df_time['Date_Clean'] = df_time['Date_Time'].astype(str).str[:14]
            df_time['True_Date'] = pd.to_datetime(df_time['Date_Clean'], format='%Y%m%d%H%M%S', errors='coerce')
            df_time = df_time.dropna(subset=['True_Date'])
            
            if len(df_time) == 0:
                st.warning("Pas de dates valides trouvées pour tracer l'évolution.")
            else:
                granularite = st.radio(" Précision de la chronologie :", options=["Moyenne par Jour", "Moyenne par Semaine", "Moyenne par Mois"], horizontal=True)
                
                if granularite == "Moyenne par Jour":
                    freq = "D"
                elif granularite == "Moyenne par Semaine":
                    freq = "W"
                else:
                    freq = "M"

                df_time['Periode'] = df_time['True_Date'].dt.to_period(freq).dt.to_timestamp()
                df_trend = df_time.groupby(['Periode', 'Machine'])['Error_pct'].mean().reset_index()
                
                st.markdown("#### Evolution de l'erreur (Dernière période vs Précédente)")
                machines_presentes = sorted(df_trend['Machine'].unique())
                
                if len(machines_presentes) > 0:
                    colonnes_kpi = st.columns(len(machines_presentes))
                    for i, mach in enumerate(machines_presentes):
                        df_mach = df_trend[df_trend['Machine'] == mach].sort_values(by='Periode')
                        if len(df_mach) >= 2:
                            current_err = df_mach.iloc[-1]['Error_pct']  
                            previous_err = df_mach.iloc[-2]['Error_pct'] 
                            delta_err = current_err - previous_err
                            
                            colonnes_kpi[i].metric(
                                label=f"Tendance {mach}",
                                value=f"{current_err:.2f} %",
                                delta=f"{delta_err:+.2f} %",
                                delta_color="inverse" 
                            )
                        else:
                            current_err = df_mach.iloc[-1]['Error_pct']
                            colonnes_kpi[i].metric(
                                label=f"Tendance {mach}",
                                value=f"{current_err:.2f} %",
                                delta="Donnée unique",
                                delta_color="off"
                            )
                st.markdown("<br>", unsafe_allow_html=True)

                fig_time = px.line(
                    df_trend,
                    x='Periode',
                    y='Error_pct',
                    color='Machine',            
                    markers=True,                
                    title=f"Évolution Temporelle de l'Erreur ({granularite})",
                    labels={'Periode': 'Date de traitement', 'Error_pct': 'Erreur Moyenne (%)', 'Machine': 'Machine Tomo'}
                )
                
                fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0))
                fig_time.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text=f"Seuil d'alerte ({seuil_alerte}%)")
                st.plotly_chart(fig_time, use_container_width=True)
        # endregion

        # region 35 - SOUS-ONGLET D : Complexité vs Erreur (Analyse Physique)
        with sub_tab_complex:
            st.markdown("<small style='color: #6c757d;'>Corrélation entre les paramètres cinématiques du plan (Vitesse de rotation, Vitesse de table) et les erreurs de transfert. <b>Les plans initiaux sont exclus.</b></small><br><br>", unsafe_allow_html=True)
                    
            param_choice = st.radio("Sélectionnez le paramètre physique en abscisse (X) :", options=["Gantry Period (Temps de rotation en s)", "Couch Speed (Vitesse de table en mm/s)"], horizontal=True)
                    
            x_col = 'GP_s' if 'Gantry' in param_choice else 'CS_mm_s'
            x_label = 'Gantry Period (s)' if 'Gantry' in param_choice else 'Couch Speed (mm/s)'
                    
            df_complex = df_stats.dropna(subset=[x_col, 'Error_pct']).copy()
                    
            df_complex['session_num'] = df_complex.groupby('Patient_ID').cumcount()
            df_complex = df_complex[df_complex['session_num'] > 0]
                    
            df_complex = df_complex[df_complex[x_col] > 0]
                    
            if len(df_complex) == 0:
                st.warning("Aucun transfert avec des données physiques valides n'est disponible pour générer le graphique.")
            else:
                df_complex = df_complex.sort_values(by=x_col)
                
                fig_complex = px.line(
                    df_complex,
                    x=x_col,
                    y='Error_pct',
                    color='Machine',            
                    markers=True,               
                    hover_data=['Patient_ID', 'Raw_Treatment_Site'], 
                    title=f"Évolution de l'Erreur en fonction du {x_label} par Machine",
                    labels={x_col: x_label, 'Error_pct': 'Erreur Moyenne (%)', 'Machine': 'Machine Tomo'},
                )
                        
                fig_complex.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=50, l=0, r=0))
                fig_complex.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text=f"Seuil d'alerte ({seuil_alerte}%)")
                        
                if x_col == 'GP_s':
                    fig_complex.add_annotation(
                        text="← Rotation rapide", xref="paper", yref="paper",
                        x=0, y=-0.15, showarrow=False,
                        font=dict(size=12, color="#7f8c8d", style="italic")
                    )
                    fig_complex.add_annotation(
                        text="Rotation lente →", xref="paper", yref="paper",
                        x=1, y=-0.15, showarrow=False,
                        font=dict(size=12, color="#7f8c8d", style="italic")
                    )
                elif x_col == 'CS_mm_s':
                    fig_complex.add_annotation(
                        text="← Table lente", xref="paper", yref="paper",
                        x=0, y=-0.15, showarrow=False,
                        font=dict(size=12, color="#7f8c8d", style="italic")
                    )
                    fig_complex.add_annotation(
                        text="Table rapide →", xref="paper", yref="paper",
                        x=1, y=-0.15, showarrow=False,
                        font=dict(size=12, color="#7f8c8d", style="italic")
                    )

                st.plotly_chart(fig_complex, use_container_width=True)
        # endregion

# region 36 - DOSSIER PATIENT
elif st.session_state.vue_actuelle == "Dossier":
    
    if st.button("⬅️ Retour au tableau de bord général"):
        st.session_state.vue_actuelle = "Accueil"
        st.session_state.patient_cible = None
        st.rerun()

    st.markdown("---")
    id_target = st.session_state.patient_cible
    
    with st.spinner("Récupération du dossier..."): 
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Isolation du retraitement : Récupération de la localisation du traitement actif le plus récent
        cursor.execute("SELECT Treatment_Site FROM SESSIONS WHERE Patient_ID = ? ORDER BY Date_Time DESC LIMIT 1", (id_target,))
        latest_site_row = cursor.fetchone()
        latest_site = latest_site_row[0] if latest_site_row else "Inconnu"
        
        # Requête ajustée pour extraire exclusivement les plans de la série active courante
        cursor.execute("""
            SELECT s.Display_Date, s.Machine, s.Dose_Gy, s.Nb_Frac, s.uLCT, s.Error_pct, s.Profile_JSON, s.Date_Time, p.Full_Name, s.CS_mm_s, s.GP_s, s.Treatment_Site, s.Raw_Treatment_Site 
            FROM SESSIONS s
            JOIN PATIENTS p ON s.Patient_ID = p.Patient_ID
            WHERE s.Patient_ID = ? AND s.Treatment_Site = ?
            ORDER BY s.Date_Time ASC
        """, (id_target, latest_site))
        
        session_lines = cursor.fetchall()
        conn.close()
        
        
        raw_data_sql = []
        for row in session_lines:
            raw_data_sql.append({
                'Date': row[0], 'Machine': row[1], 'Dose (Gy)': row[2], 'Nb_Frac': row[3],
                'uLCT (%)': row[4], 'Session Error (%)': row[5], 'Profile_Error': json.loads(row[6]),
                'sort_key': row[7], 'Patient': row[8], 'ID': id_target,
                'CS': row[9], 'GP': row[10], 'Site': row[11], 'Raw_Site': row[12]
            })
            
        display_dashboard(raw_data_sql)
# endregion
