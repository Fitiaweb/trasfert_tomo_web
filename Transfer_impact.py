# region 1 - Importation des bibliothèques
import re                         # Pour la détection intelligente de texte (Regex)
import pydicom as dcm                # Pour lire et manipuler les fichiers médicaux DICOM
import numpy as np                # Pour les calculs mathématiques et la gestion des matrices
import pandas as pd               # Pour manipuler les tableaux de données
import os                         # Pour interagir avec le système de fichiers
import shutil                     # Pour déplacer des fichiers (archivage)
import matplotlib.pyplot as plt   # Pour tracer le graphique polaire
import time                       # Pour ajouter des petites pauses
import sqlite3                    # Pour gérer la base de données locale SQL
import json                       # Pour stocker les matrices complexes en texte
import plotly.express as px       # Pour créer des graphiques statistiques interactifs
import streamlit as st            # Bibliothèque pour créer l'interface web interactive
import math                       # Pour les calculs d'arrondis stricts
from fpdf import FPDF             # Pour générer le rapport PDF
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
        Fractions_Done INTEGER DEFAULT -1,
        FOREIGN KEY (Patient_ID) REFERENCES PATIENTS(Patient_ID)
    )
    ''')
    
    # Mise a jour securisee si la table existe deja mais sans la colonne Fractions_Done
    try:
        cursor.execute("ALTER TABLE SESSIONS ADD COLUMN Fractions_Done INTEGER DEFAULT -1")
    except sqlite3.OperationalError:
        pass 

    conn.commit()  # Valide les modifications
    conn.close()   # Ferme la connexion proprement

init_db() # Appelle la fonction pour s'assurer que la BDD est prête au lancement
# endregion

def harmoniser_base_de_donnees():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE SESSIONS SET Treatment_Site = 'Hémato' WHERE Treatment_Site = 'Hémato & Ganglionnaire'")
    cursor.execute("UPDATE SESSIONS SET Machine = 'Ancienne Tomo4' WHERE Machine IN ('210462', 'Tomo4')")
    cursor.execute("UPDATE SESSIONS SET Treatment_Site = 'TBI' WHERE Raw_Treatment_Site LIKE '%FF%' OR Raw_Treatment_Site LIKE '%HF%' OR Raw_Treatment_Site LIKE '%TBI%'")
    cursor.execute("UPDATE SESSIONS SET Treatment_Site = 'CSI' WHERE Raw_Treatment_Site LIKE '%CSI%' OR Raw_Treatment_Site LIKE '%cranio%' OR Raw_Treatment_Site LIKE '%spinal%'")
    
    # NOUVEAU : Nettoyage plus large pour toutes les façons d'écrire R1 et R2
    cursor.execute("""
        UPDATE SESSIONS SET Treatment_Site = Treatment_Site || ' - R1' 
        WHERE (Raw_Treatment_Site LIKE '%_R1%' OR Raw_Treatment_Site LIKE '%_R01%' OR Raw_Treatment_Site LIKE '%_R_01%' OR Raw_Treatment_Site LIKE '%_R_1%') 
        AND Treatment_Site NOT LIKE '%- R1%'
    """)
    cursor.execute("""
        UPDATE SESSIONS SET Treatment_Site = Treatment_Site || ' - R2' 
        WHERE (Raw_Treatment_Site LIKE '%_R2%' OR Raw_Treatment_Site LIKE '%_R02%' OR Raw_Treatment_Site LIKE '%_R_02%' OR Raw_Treatment_Site LIKE '%_R_2%') 
        AND Treatment_Site NOT LIKE '%- R2%'
    """)
    conn.commit()
    conn.close()
# Appel à ajouter une fois après l'init_db()
harmoniser_base_de_donnees()

# region 5 - Catégorisation Dynamique
def load_categories(filepath="categories.txt"):
    """Charge le dictionnaire des localisations anatomiques depuis un fichier texte externe."""
    categories = {}
    if os.path.exists(filepath):
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
@st.cache_data
def charger_donnees_globales():
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT Patient_ID, Date_Time, Machine, Treatment_Site, Raw_Treatment_Site, Error_pct, GP_s, CS_mm_s FROM SESSIONS"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Nettoyage automatique des noms de localisation pour les statistiques
    if not df.empty and 'Treatment_Site' in df.columns:
        df['Treatment_Site'] = df['Treatment_Site'].astype(str).str.replace('Hémato & Ganglionnaire', 'Hémato', regex=False)
        
    return df
# endregion

# region 7 - Sinogram Extraction
def get_sinogram(plan):
    """Extrait la matrice d'ouverture des lames (sinogramme) à partir du fichier RTPLAN."""
    NCP = plan.BeamSequence[0].NumberOfControlPoints  # Récupère le nombre de points de contrôle
    sinogram = np.zeros((NCP, 64))                    # Initialise une matrice vide pour le sinogramme
    cp_sequence = plan.BeamSequence[0].ControlPointSequence 

    for cp in range(NCP):                               # Boucle sur chaque point de contrôle
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
    plan_info["patient_id"] = str(plan.PatientID)      # Extrait l'ID du patient
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
    texte_minuscule = texte_brut.lower()    
    plan_info["treatment_site"] = "Inconnu" 
    
    if any(mot in texte_minuscule for mot in ["_ff", "_hf", "ff_", "hf_", "tbi", "total body"]):
        plan_info["treatment_site"] = "TBI"
    elif any(mot in texte_minuscule for mot in ["csi", "cranio", "spinal"]):
        plan_info["treatment_site"] = "CSI"
    else:
        for category, keywords in CATEGORIES_DICT.items():
            if any(mot in texte_minuscule for mot in keywords):
                plan_info["treatment_site"] = category
                break                              
                
    # 3. Détection intelligente des replanifications (R1, R_01, R01, r_1...)
    # Cherche _, - ou espace, suivi de 'r', optionnellement un '_', optionnellement un '0', puis un chiffre de 1 à 9
    match = re.search(r"[-_ ]r_?0?([1-9])", texte_minuscule)
    if match:
        num_replan = match.group(1) # Extrait juste le chiffre pur (ex: "1" à partir de "_r_01")
        plan_info["treatment_site"] += f" - R{num_replan}"
    # endregion

    # region 10 - mapping
    # Association entre le numéro de série physique de l'appareil et son nom d'usage dans le service
    serial_mapping = {
        "4010012": "Tomo2", 
        "4010710": "Radi7",
        "210462": "Ancienne Tomo4" # Géré pour l'historique ou les exports par erreur
    } 
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
    delivery["PT"] = (delivery["GP"] / 51.0) * 1000.0                        # Temps par projection (ms)
    delivery["CS"] = float(plan.BeamSequence[0][0x300d, 0x1080].value)   # Vitesse de table (Couch Speed en mm/s)
    delivery["pitch"] = float(plan.BeamSequence[0][0x300d, 0x1060].value) # Facteur de pitch
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    delivery["Nrot"] = (NCP - 1) / 51                                      # Nombre de rotations totales
    delivery["TT"] = delivery["Nrot"] * delivery["GP"]                     # Temps total de traitement
    
    try:
        delivery["DS"] = float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose) # Dose par fraction
        delivery["Nb_Frac"] = int(plan.FractionGroupSequence[0].NumberOfFractionsPlanned)        # Nombre de séances
    except Exception:
        delivery["DS"] = 0.0
        delivery["Nb_Frac"] = 1
    return delivery
# endregion

# region 12 - Error Calculation et Calcul Strict
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
    
    for i in range(len(filtered_row) - 1):
        diff = (PT - LOT_sino[filtered_row[i], filtered_col[i]]) 
        extra_time += diff  
        error_per_projection[filtered_row[i]] += diff # Cumul des temps d'erreur en ms
    
    return ((extra_time / total_lot) * 100), ((undisc_LCT / (len(open_leaves_LOT))) * 100), error_per_projection

def get_max_sessions_strictly_inferior(budget, current_dose, remain, dose_nom, dose_err):
    """Logique mathématique stricte pour calculer le nombre maximal de séances, en respectant l'affichage à 2 décimales."""
    if remain <= 0: return 0
    if dose_err <= dose_nom: return remain
    
    # On force la valeur affichée pour être WYSIWYG (What You See Is What You Get)
    budget_limit = float(f"{budget:.2f}")
    
    # On teste en descendant, de 'remain' (le maximum possible) jusqu'à 0
    for n in range(remain, -1, -1):
        simulated_dose = current_dose + (n * dose_err) + ((remain - n) * dose_nom)
        # On compare la version strictement arrondie à 2 décimales pour correspondre à l'écran
        if float(f"{simulated_dose:.2f}") < budget_limit:
            return n
            
    return 0
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
    """Génère le document PDF récapitulatif avec affichage du statut final."""
    import unicodedata
    
    def clean_txt(t):
        """Supprime les accents et nettoie le texte pour FPDF."""
        if pd.isna(t): return ""
        # Nettoyage des balises HTML utilisées dans Streamlit
        t = str(t).replace("<b>", "").replace("</b>", "").replace("<br>", " ").replace("<i>", "").replace("</i>", "")
        # Normalisation pour enlever les accents
        nfkd = unicodedata.normalize('NFKD', t)
        return "".join([c for c in nfkd if not unicodedata.combining(c)]).encode('ascii', 'replace').decode('ascii')

    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, clean_txt("Rapport de Transfert TomoTherapy"), ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, clean_txt("Informations Patient"), ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 6, clean_txt(f"Nom : {patient_name}"), ln=True)
    pdf.cell(0, 6, clean_txt(f"ID : {patient_id}"), ln=True)
    pdf.cell(0, 6, clean_txt(f"Localisation : {site}"), ln=True)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, clean_txt("Bilan Dosimetrique Actuel"), ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 6, clean_txt(f"Dose Cumulee Estimee : {current_dose:.2f} Gy"), ln=True)
    pdf.cell(0, 6, clean_txt(f"Dose maximum toleree : {budget_max:.2f} Gy"), ln=True)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, clean_txt("Historique des Seances"), ln=True)
    
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(22, 8, clean_txt("Date"), border=1)
    pdf.cell(20, 8, clean_txt("Machine"), border=1)
    pdf.cell(15, 8, clean_txt("Frac."), border=1)
    pdf.cell(20, 8, clean_txt("Err(%)"), border=1)
    pdf.cell(32, 8, clean_txt("Dose/seance"), border=1)
    pdf.cell(32, 8, clean_txt("Dose Etape"), border=1)
    pdf.ln()
    
    pdf.set_font("helvetica", "", 9)
    for index, row in df_history.iterrows():
        pdf.cell(22, 8, clean_txt(str(row['Date'])[:10]), border=1)
        pdf.cell(20, 8, clean_txt(str(row['Machine'])), border=1)
        pdf.cell(15, 8, clean_txt(str(row['Fractions Réalisées'])), border=1)
        pdf.cell(20, 8, clean_txt(str(row['Erreur Séance (%)'])), border=1)
        pdf.cell(32, 8, clean_txt(str(row['Dose Délivrée (Gy/séance)'])), border=1)
        pdf.cell(32, 8, clean_txt(str(row['Dose Totale Étape (Gy)'])), border=1)
        pdf.ln()
        
    pdf.ln(10)

    # Affichage du statut de fin de traitement (toujours affiché)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, clean_txt("Statut de fin de traitement :"), ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, clean_txt(alerte_message))
    
    return bytes(pdf.output())
# endregion



# region 15 - Dashboard Engine
def display_dashboard(raw_data):
    """Génère l'affichage complet du dossier clinique du patient sélectionné."""
    if not raw_data:
        st.warning("Aucune donnée à afficher pour ce patient.")
        return

    patient_name = raw_data[0]['Patient']
    patient_id = raw_data[0]['ID']
    nb_frac_ref = raw_data[0]['Nb_Frac']
    site_traitement = raw_data[0].get('Site', 'Inconnu')  
    site_brut = raw_data[0].get('Raw_Site', 'Inconnu')   
    
    html_header = f"""
    <div style="background-color: #f8f9fa; padding: 15px 25px; border-radius: 8px; border-left: 6px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Dossier Patient - Localisation : <span style="color: #d32f2f;">{site_traitement}</span></span>
            <h2 style="margin: 5px 0 0 0; color: #2c3e50; font-size: 26px;">{patient_name}</h2>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Séances prévues</span>
            <h3 style="margin: 5px 0 0 0; color: #2c3e50; font-size: 24px; font-weight: 700;">{nb_frac_ref}</h3>
        </div>
        <div style="text-align: center; border-left: 2px solid #e9ecef; padding-left: 20px;">
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Nom du plan </span>
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
        
        liste_sites_propres = list(CATEGORIES_DICT.keys()) + ["TBI", "CSI", "Autre", "Inconnu"]

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
                cursor_update.execute("UPDATE SESSIONS SET Treatment_Site = ? WHERE Patient_ID = ?", (nouveau_site, patient_id))
                conn_update.commit()
                conn_update.close()
                time.sleep(0.5) 
                st.rerun() 
    st.markdown("<br>", unsafe_allow_html=True)
    # endregion

    # region 17 - Saisie des fractions (Validation manuelle sans enregistrement BDD)
    st.markdown("##### Validation des séances réalisées")
    st.markdown("<small style='color: #6c757d;'>Le DICOM ne reflétant pas toujours la réalité clinique, saisissez le nombre de séances effectuées pour la simulation.</small>", unsafe_allow_html=True)
    
    fractions_done_list = []
    cols = st.columns(len(raw_data))
    for index in range(len(raw_data)):
        machine = raw_data[index]['Machine']
        with cols[index]:
            # Initialisation par défaut à 0 pour toutes les machines (sécurité clinique)
            f_done = st.number_input(f" {machine}", min_value=0, max_value=int(nb_frac_ref), value=0, key=f"frac_{index}")
            fractions_done_list.append(f_done)
    st.markdown("<br>", unsafe_allow_html=True)
    # endregion

    # region 18 - CALCUL DU BUDGET DOSE
    seuil_patient = st.session_state.get("seuil_global_val", 1.5)
    total_cumulated_dose = 0.0 
    final_table_data = [] 
    
    try:
        dose_nominal_ref = float(raw_data[0]['Dose (Gy)'])
        dose_totale_prescrite = dose_nominal_ref * nb_frac_ref
        total_budget_Gy = dose_totale_prescrite * (1 + (seuil_patient / 100.0))
    except:
        total_budget_Gy = 0.0
        dose_totale_prescrite = 0.0

    for index, item in enumerate(raw_data):
        try:
            error_session_pct = float(item['Session Error (%)'])
        except (TypeError, ValueError):
            error_session_pct = 0.0
            
        dose_nominal = float(item['Dose (Gy)'])
        fractions_done = fractions_done_list[index]
        
        actual_dose_session = dose_nominal * (1 + (error_session_pct / 100.0)) if index > 0 else dose_nominal
        dose_totale_etape = actual_dose_session * fractions_done
        total_cumulated_dose += dose_totale_etape
        
        if index == 0:
            date_display = item['Date'] + " (Initiale)" 
            comment = f"Prescription : {dose_totale_prescrite:.2f} Gy"
            alert = False
        else:
            date_display = item['Date']
            alert = error_session_pct > seuil_patient
            comment = f"Erreur > {seuil_patient}%" if alert else "OK"
        
        final_table_data.append({
            "Date": date_display, 
            "Machine": item['Machine'],
            "Fractions Réalisées": fractions_done,
            "Dose Prescrite (Gy/séance)": f"{dose_nominal:.2f}", 
            "Erreur Séance (%)": f"{error_session_pct:.2f}" if index > 0 else "-",
            "Dose Délivrée (Gy/séance)": f"{actual_dose_session:.2f}" if index > 0 else f"{dose_nominal:.2f}", 
            "Dose Totale Étape (Gy)": f"{dose_totale_etape:.2f}",
            "Dose Cumulée (Gy)": f"{total_cumulated_dose:.2f}", 
            "Commentaire": comment, 
            "_Alert": alert, "_Index": index, "_Cumul_Val": total_cumulated_dose, "_Budget_Total": total_budget_Gy, "_Nb_Frac_Plan": item['Nb_Frac'],
            "_Dose_Nom_Exact": dose_nominal, "_Dose_Err_Exact": actual_dose_session
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
                elif row['Machine'] == 'Radi7': cell_style = 'background-color: #c8e6c9; color: #000000; font-weight: bold;' 
                # Style gris pour l'ancienne machine
                elif row['Machine'] in ['Tomo4', 'Ancienne Tomo4']: cell_style = 'background-color: #e0e0e0; color: #6c757d; font-style: italic;' 
            styles.append(cell_style)
        return styles
    # endregion

    # region 22 - Affichage du tableau final
    
    # 1. On applique le style directement sur le DataFrame complet 
    # (qui contient bien _Index et _Alert pour que la fonction marche)
    styled_df = df.style.apply(style_dataframe, axis=1)
    
    # 2. On affiche le tableau en masquant les colonnes techniques via column_config
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "_Alert": None, 
            "_Index": None, 
            "_Cumul_Val": None, 
            "_Budget_Total": None, 
            "_Nb_Frac_Plan": None, 
            "_Dose_Nom_Exact": None, 
            "_Dose_Err_Exact": None
        }
    )
    st.markdown("---")
    # endregion

    col_graph1, col_graph2 = st.columns([1, 1]) 

        # region 23 - Bloc de gauche : PRÉVISION DE FIN DE TRAITEMENT
    with col_graph1:
        alerte_message = "Traitement en cours ou terminé."  # valeur par défaut pour le PDF

        if len(df) > 0:
            last_session_df = df.iloc[-1]
            budget_max = last_session_df['_Budget_Total']
            current_dose = last_session_df['_Cumul_Val']

            if len(df) > 1:
                machine_actuelle = last_session_df['Machine']
                machine_initiale = raw_data[0]['Machine']  # Récupère la machine d'origine du plan initial
                dose_nom = float(last_session_df['_Dose_Nom_Exact'])
                dose_err = float(last_session_df['_Dose_Err_Exact'])

                total_done_simul = sum(fractions_done_list)
                remain = nb_frac_ref - total_done_simul

                max_sessions_possibles = get_max_sessions_strictly_inferior(
                    budget_max, current_dose, remain, dose_nom, dose_err
                )

                if remain > 0:
                    # Calcul de la répartition exacte des séances restantes
                    if dose_err > dose_nom:
                        if max_sessions_possibles == remain:
                            seances_machine_actuelle = remain
                            seances_machine_initiale = 0
                        else:
                            transfer_back = remain - max_sessions_possibles
                            seances_machine_actuelle = max_sessions_possibles
                            seances_machine_initiale = transfer_back
                    else:
                        seances_machine_actuelle = remain
                        seances_machine_initiale = 0

                    # Texte utilisé pour le PDF
                    alerte_message = (
                        f"Commentaire : Le patient peut faire {seances_machine_actuelle} séance(s) "
                        f"sur {machine_actuelle} et {seances_machine_initiale} séance(s) restante(s) "
                        f"doivent être faites sur {machine_initiale}."
                    )

                    # Affichage des deux badges de répartition
                    st.markdown(
                        "<h5 style='font-size: 15px; color: #495057; font-weight: 600; margin-bottom: 10px;'>Planification des séances restantes :</h5>",
                        unsafe_allow_html=True
                    )
                    col_cards1, col_cards2 = st.columns(2)

                    with col_cards2:
                        color_actuelle = "#c62828" if (dose_err > dose_nom and max_sessions_possibles < remain) else "#2e7d32"
                        bg_actuelle = "#ffebee" if (dose_err > dose_nom and max_sessions_possibles < remain) else "#e8f5e9"
                        st.markdown(f"""
                        <div style="background-color: {bg_actuelle}; border: 2px solid {color_actuelle}; border-radius: 8px; padding: 15px; text-align: center;">
                            <span style="font-size: 11px; color: #555; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Sur Machine Transfert ({machine_actuelle})</span>
                            <h2 style="margin: 5px 0 0 0; color: {color_actuelle}; font-size: 32px; font-weight: 800;">{seances_machine_actuelle} <span style='font-size: 18px; font-weight: normal;'>séance(s)</span></h2>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_cards1:
                        color_initiale = "#1565c0" if seances_machine_initiale > 0 else "#6c757d"
                        bg_initiale = "#e3f2fd" if seances_machine_initiale > 0 else "#f8f9fa"
                        st.markdown(f"""
                        <div style="background-color: {bg_initiale}; border: 2px solid {color_initiale}; border-radius: 8px; padding: 15px; text-align: center;">
                            <span style="font-size: 11px; color: #555; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">Retour Machine Origine ({machine_initiale})</span>
                            <h2 style="margin: 5px 0 0 0; color: {color_initiale}; font-size: 32px; font-weight: 800;">{seances_machine_initiale} <span style='font-size: 18px; font-weight: normal;'>séance(s)</span></h2>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    if float(f"{current_dose:.2f}") >= float(f"{budget_max:.2f}"):
                        alerte_message = f"ALERTE SURDOSE : Le traitement est terminé mais la dose totale ({current_dose:.2f} Gy) atteint ou dépasse le budget limite de {budget_max:.2f} Gy."
                        st.error(alerte_message)
                    else:
                        alerte_message = "Le traitement est théoriquement terminé et conforme."
                        st.success(alerte_message)
            else:
                alerte_message = f"Le patient n'a subi aucun transfert. Il reste {nb_frac_ref - sum(fractions_done_list)} séances prévues sur la machine d'origine."
                st.info(alerte_message)

            st.markdown("<br><hr><br>", unsafe_allow_html=True)
    # endregion

    # region 24 - Bloc de gauche : BILAN DOSIMÉTRIQUE CUMULÉ
            st.subheader("Bilan Dosimétrique Cumulé") 

            # Calcul décimal du pourcentage pour la jauge progress (1.0 = 100%)
            percentage = min(current_dose / budget_max, 1.0) if budget_max > 0 else 0.0
            
            # Logique WYSIWYG : on passe en alerte rouge UNIQUEMENT SI c'est strictement inférieur (exclut l'égalité visuelle)
            dose_cumulee_val = float(f"{current_dose:.2f}")
            budget_max_val = float(f"{budget_max:.2f}")
            
            est_strictement_inferieur = dose_cumulee_val < budget_max_val
            
            bg_color = "#e8f5e9" if est_strictement_inferieur else "#ffebee"
            text_color = "#2e7d32" if est_strictement_inferieur else "#c62828"
            
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
                label="Exporter le Rapport Clinique (PDF)",
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
            st.info("Aucun transfert machine détecté pour ce patient.")
        else:
            menu_options = []
            for s in transfer_sessions:
                menu_options.append(f"Transfer: {s['Date']} to {s['Machine']}")
                    
            selected_date = st.selectbox("Select the transfer event to analyze:", options=menu_options)
            chosen_index = menu_options.index(selected_date)
            session_to_analyze = transfer_sessions[chosen_index]

            # Conversion en float pour la sécurité
            try: session_err = float(session_to_analyze['Session Error (%)'])
            except: session_err = 0.0

            if session_err > 0.0:
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

                    selection = st.select_slider("Avancement de la table sur l'axe longitudinal (Gz)", options=options_cm) 
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
                
                # Grille sur 51 projections (Toutes affichées)
                angles_proj = np.linspace(0, 2 * np.pi, 51, endpoint=False)
                ax.set_xticks(angles_proj) 
                
                # Création des labels de 1 à 51
                labels_proj = [str(i + 1) for i in range(51)]
                        
                ax.set_xticklabels(labels_proj, fontsize=6, color='#2c3e50') 

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
                st.info("Aucune erreur détectée pour cette session.")
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
            label="Télécharger la Procédure",
            data=file,
            file_name="Procédure_Tomo_Transfer.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
except FileNotFoundError:
    st.sidebar.warning("Fichier de procédure introuvable sur le réseau.")
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
                pass
            
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
        time.sleep(1)
        st.rerun() 
else:
    st.sidebar.success("Base de données à jour. En attente de nouveaux DICOM...")
# endregion

# region 30 - GESTION DE LA NAVIGATION (passer de l'accueil au dossier patient)
if "vue_actuelle" not in st.session_state:
    st.session_state.vue_actuelle = "Accueil" 
if "patient_cible" not in st.session_state:
    st.session_state.patient_cible = None
if "site_cible" not in st.session_state:
    st.session_state.site_cible = None
# endregion

# region 31 - MENU DE NAVIGATION Accueil & Statistiques
if st.session_state.vue_actuelle == "Accueil":
    st.markdown("### Suivi transfert patient")
    
    seuil_alerte = st.number_input(
    "Définir le seuil d'alerte clinique par transfert (%) :", 
    min_value=0.0, max_value=10.0, value=1.5, step=0.1, key="seuil_global",
    help="Seuil de tolérance pour une séance individuelle."
    )
    seuil_alerte = round(seuil_alerte, 2)  # <-- sécurise la valeur contre les erreurs de float
    st.session_state["seuil_global_val"] = seuil_alerte
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
                site = row[6] # Récupère la localisation (ex: ORL - R1)
                cle_dossier = f"{pid}_{site}" # Clé unique !
                
                if cle_dossier not in patients_dict: patients_dict[cle_dossier] = []
                patients_dict[cle_dossier].append(row)
            
            tableau_final = []
            
            for cle_dossier, sessions in patients_dict.items():
                plan_initial = sessions[0]
                plan_actuel = sessions[-1] 
                
                pid = plan_actuel[0]
                nom_patient = plan_actuel[1]
                machine_actuelle = plan_actuel[2]
                dose_actuelle = float(plan_actuel[3])
                erreur_actuelle = float(plan_actuel[5])
                site = str(plan_actuel[6]).replace('Hémato & Ganglionnaire', 'Hémato')
                
                nb_frac_ref = plan_initial[4]
                
                if len(sessions) > 1:
                    dose_init = float(plan_initial[3])
                    budget_theorique = dose_init * nb_frac_ref * (1 + (seuil_alerte / 100.0))
                    
                    current_dose_sim = 0.0
                    fractions_accumulated_sim = 0
                    
                    for i in range(len(sessions) - 1):
                        # Calcul dynamique par différence DICOM (diff_auto)
                        frac = int(max(0, sessions[i][4] - sessions[i+1][4]))
                        idx_dose = float(sessions[i][3])
                        idx_err = float(sessions[i][5])
                        act_dose = idx_dose * (1 + (idx_err / 100.0)) if i > 0 else idx_dose
                        current_dose_sim += act_dose * frac
                        fractions_accumulated_sim += frac
                    
                    actual_last_dose_per_frac = dose_actuelle * (1 + (erreur_actuelle / 100.0))
                    remain_sim = nb_frac_ref - fractions_accumulated_sim
                    
                    max_sessions = get_max_sessions_strictly_inferior(budget_theorique, current_dose_sim, remain_sim, dose_init, actual_last_dose_per_frac)
                    
                    # Alerte si erreur > seuil OU si on doit transférer des séances par manque de budget
                    machine_initiale = plan_initial[2]
                    alerte_erreur = erreur_actuelle > seuil_alerte
                    
                    # Fusion des colonnes Séances Max et Statut
                    if alerte_erreur or max_sessions < remain_sim:
                        statut = f"Alerte : {max_sessions} séance(s) max sur {machine_actuelle}"
                    else:
                        statut = f"Conforme : {remain_sim} séance(s) sur {machine_actuelle}"
                    
                    erreur_str = f"{erreur_actuelle:.2f} %"
                else: 
                    statut = "Plan Initial"
                    erreur_str = "-"
                
                tableau_final.append({
                    "Date (Dernier import)": plan_actuel[7],
                    "ID Patient": pid,
                    "Nom": nom_patient,
                    "Localisation": site,
                    "Machine initiale": plan_initial[2],
                    "Erreur Transfert": erreur_str,
                    "Statut": statut
                })
            
            df_recap = pd.DataFrame(tableau_final)
            
            # Application de l'ordre de tes colonnes
            ordre_recap = [
                "Date (Dernier import)", "ID Patient", "Nom", 
                "Statut", "Localisation", "Machine initiale", "Erreur Transfert"
            ]
            df_recap = df_recap[[col for col in ordre_recap if col in df_recap.columns]]
            
            def colorer_statut(row):
                styles = [''] * len(row)
                statut = row['Statut']
                if "Alerte" in statut:
                    styles[row.index.get_loc('Statut')] = 'background-color: #c62828; color: white; font-weight: bold; text-align: center;'
                elif "Conforme" in statut:
                    styles[row.index.get_loc('Statut')] = 'background-color: #2e7d32; color: white; font-weight: bold; text-align: center;'
                return styles
            
            st.info("**Navigation :** Cochez la case à gauche pour ouvrir le dossier.")
            
            event = st.dataframe(
                df_recap.style.apply(colorer_statut, axis=1), 
                use_container_width=True, 
                hide_index=True, 
                on_select="rerun", 
                selection_mode="single-row",
                column_config={
                    "Statut": st.column_config.TextColumn(
                        "Statut de fin de traitement",
                        width=450,
                    )
                }
            )
            
            if event.selection.rows:
                row_selected = df_recap.iloc[event.selection.rows[0]]
                st.session_state.patient_cible = row_selected["ID Patient"]
                st.session_state.site_cible = row_selected["Localisation"]
                st.session_state.vue_actuelle = "Dossier"
                st.rerun() 
        
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### Statistiques Globales")

        df_stats = df_stats[~df_stats['Machine'].isin(['Tomo4', 'Ancienne Tomo4', '210462'])]
        # ------------------------------------------------------------------------------------

        sub_tab_loc, sub_tab_mach, sub_tab_time = st.tabs(["Par Localisation", "Par Sens de Transfert", "Évolution dans le Temps"])
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

                 # 1. CRÉATION DU TEXTE PERSONNALISÉ
                 df_mean_loc['Texte_Affichage'] = "MOYENNE : " + df_mean_loc['Error_pct'].round(2).astype(str) + " %"

                 # 2. MODIFICATION DU GRAPHIQUE
                 fig_loc = px.bar(
                     df_mean_loc,
                     x='Treatment_Site',
                     y='Error_pct',
                     title="Erreur Moyenne par catégorie de localisation",
                     labels={'Treatment_Site': 'Localisation Anatomique', 'Error_pct': 'Erreur Moyenne (%)'},
                     text='Texte_Affichage',
                     color='Error_pct', 
                     color_continuous_scale='Reds' 
                 )

                 fig_loc.update_layout(xaxis_tickangle=-45, plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0), font=dict(size=14))
                
                 # 3. MISE EN FORME DU TEXTE DANS LA BARRE + NETTOYAGE DU SURVOL
                 fig_loc.update_traces(
                     textfont_size=24,
                     textposition='inside',
                     insidetextanchor='middle',
                     textfont_color='white',
                     textfont_weight='bold',
                     hovertemplate="<b>%{x}</b><br>Erreur Moyenne : %{y:.2f} %<extra></extra>"
                 )
                
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
                trajet_choisi = st.selectbox("Filtrer les compteurs et la répartition pour un trajet spécifique :", ["Tous les transferts (Global)"] + liste_trajets)
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
                col_kpi1.metric("Total de Transferts", total_focus)
                col_kpi2.metric(f"Transferts > {seuil_alerte :.2f}% (Alerte)", int(depassements_focus))
                col_kpi3.metric("Taux d'alerte", f"{taux_focus:.1f} %")
                
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
                        title="Erreur Moyenne par Trajet (Moyenne Globale)",
                        labels={
                            'Trajet': 'Sens du transfert', 
                            'Error_mean': 'Erreur Moyenne (%)',
                            'Depassement_rate': f"Taux d'alerte (>{seuil_alerte:.2f}%)",
                            'Count': 'Nombre de patients'
                        },
                        text_auto='.2f', 
                        color='Error_mean', 
                        color_continuous_scale='Oranges',
                        hover_data={'Depassement_rate': ':.1f', 'Count': True}
                    )
                    
                    fig_mach.update_traces(hovertemplate='<b>%{x}</b><br>Erreur Moyenne: %{y:.2f}%<br>Patients en alerte: %{customdata[0]:.1f}%<br>Nb total de patients: %{customdata[1]}<extra></extra>')
                    fig_mach.update_layout(xaxis_tickangle=0, plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=0, l=0, r=0), font=dict(size=14))
                    fig_mach.update_traces(textfont_size=16, textposition='outside', textfont_weight='bold')
                    fig_mach.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text=f"Seuil ({seuil_alerte:.2f}%)")
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
                        fig_pie.update_layout(margin=dict(t=40, b=0, l=0, r=0), showlegend=False, font=dict(size=14))
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
                granularite = st.radio("Précision de la chronologie :", options=["Moyenne par Jour", "Moyenne par Semaine", "Moyenne par Mois"], horizontal=True)
                
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
                    title=f"Évolution Temporelle de l'Erreur ({granularite}) - (Moyenne)",
                    labels={'Periode': 'Date de traitement', 'Error_pct': 'Erreur Moyenne (%)', 'Machine': 'Machine Tomo'}
                )
                
                fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=0, l=0, r=0), font=dict(size=14))
                fig_time.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text=f"Seuil d'alerte ({seuil_alerte}%)")
                st.plotly_chart(fig_time, use_container_width=True)
        # endregion

        

# region 36 - DOSSIER PATIENT
elif st.session_state.vue_actuelle == "Dossier":
    
    if st.button("Retour au tableau de bord général"):
        st.session_state.vue_actuelle = "Accueil"
        st.session_state.patient_cible = None
        st.session_state.site_cible = None
        st.rerun()

    st.markdown("---")
    id_target = st.session_state.patient_cible
    site_target = st.session_state.get("site_cible", None)
    
    with st.spinner("Récupération du dossier..."): 
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # On utilise le site cliqué par l'utilisateur
        if site_target:
            latest_site = site_target
        else:
            cursor.execute("SELECT Treatment_Site FROM SESSIONS WHERE Patient_ID = ? ORDER BY Date_Time DESC LIMIT 1", (id_target,))
            latest_site_row = cursor.fetchone()
            latest_site = latest_site_row[0] if latest_site_row else "Inconnu"
        
        cursor.execute("""
            SELECT s.Session_ID, s.Display_Date, s.Machine, s.Dose_Gy, s.Nb_Frac, s.uLCT, s.Error_pct, s.Profile_JSON, s.Date_Time, p.Full_Name, s.CS_mm_s, s.GP_s, s.Treatment_Site, s.Raw_Treatment_Site, s.Fractions_Done 
            FROM SESSIONS s
            JOIN PATIENTS p ON s.Patient_ID = p.Patient_ID
            WHERE s.Patient_ID = ? AND s.Treatment_Site = ?
            ORDER BY s.Date_Time ASC
        """, (id_target, latest_site))
        
        session_lines = cursor.fetchall()
        conn.close()
        
        raw_data_sql = []
        for row in session_lines:
            row_dict = {
                'Session_ID': row[0],
                'Date': row[1], 
                'Machine': row[2], 
                'Dose (Gy)': row[3], 
                'Nb_Frac': row[4],
                'uLCT (%)': row[5], 
                'Session Error (%)': row[6], 
                'Profile_Error': json.loads(row[7]),
                'sort_key': row[8], 
                'Patient': row[9], 
                'ID': id_target,
                'CS': row[10], 
                'GP': row[11], 
                'Site': str(row[12]).replace('Hémato & Ganglionnaire', 'Hémato'), 
                'Raw_Site': row[13],
                'Fractions_Done': row[14]
            }
            
            # --- FUSION DES TBI/CSI (FF + HF ou plans multiples) ---
            # Si on a déjà une ligne ET que c'est la même machine ET le même jour (les 8 premiers caractères de sort_key = YYYYMMDD)
            if len(raw_data_sql) > 0 and raw_data_sql[-1]['Machine'] == row_dict['Machine'] and raw_data_sql[-1]['sort_key'][:8] == row_dict['sort_key'][:8]:
                
                # 1. On moyenne l'erreur des plans fusionnés
                prev_err = float(raw_data_sql[-1]['Session Error (%)'])
                curr_err = float(row_dict['Session Error (%)'])
                raw_data_sql[-1]['Session Error (%)'] = (prev_err + curr_err) / 2.0
                
                # 2. On combine les noms bruts
                if row_dict['Raw_Site'] not in raw_data_sql[-1]['Raw_Site']:
                    raw_data_sql[-1]['Raw_Site'] += f" + {row_dict['Raw_Site']}"
                    
                # 3. On rallonge le profil d'erreur graphique
                raw_data_sql[-1]['Profile_Error'].extend(row_dict['Profile_Error'])
                
            else:
                raw_data_sql.append(row_dict)
                
        display_dashboard(raw_data_sql)
# endregion
