#region 1-Importation des bibliothèques
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
#endregion

#======= Page Configuration ================================================================================
# Configure le titre de l'onglet du navigateur et utilise toute la largeur de l'écran
st.set_page_config(page_title="Tomo Transfer", layout="wide")
#===========================================================================================================

#===========================================================================================================
#======= Automatic Folder & Database Setup =================================================================
DIR_IN = r"\\nasdata1\TOMO\Transfert_tomo"  # Chemin du dossier où les nouveaux DICOM sont déposés
DIR_ARCHIVE = "ARCHIVES"                    # Dossier où les DICOM sont déplacés après traitement
DB_NAME = "tomo_database.db"                # Nom du fichier de la base de données SQLite

os.makedirs(DIR_IN, exist_ok=True)          # Crée le dossier d'entrée s'il n'existe pas
os.makedirs(DIR_ARCHIVE, exist_ok=True)     # Crée le dossier d'archive s'il n'existe pas

def init_db():
    # Se connecte (ou crée) la base de données
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
#===========================================================================================================

#======= Catégorisation Dynamique ==========================================================================
def load_categories(filepath="categories.txt"):
   
    # Lit le fichier s'il existe déjà
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

#===========================================================================================================
#======= Données Globales (CACHE) ==========================================================================
@st.cache_data
def charger_donnees_globales():
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT Patient_ID, Date_Time, Machine, Treatment_Site, Raw_Treatment_Site, Error_pct, GP_s, CS_mm_s FROM SESSIONS"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
#===========================================================================================================

#======= Sinogram Extraction ===============================================================================
def get_sinogram(plan):
    NCP = plan.BeamSequence[0].NumberOfControlPoints  # Récupère le nombre de points de contrôle
    sinogram = np.zeros((NCP,64))                     # Initialise une matrice vide pour le sinogramme
    cp_sequence = plan.BeamSequence[0].ControlPointSequence 

    for cp in range(NCP):                             # Boucle sur chaque point de contrôle
        try : 
            tmp = cp_sequence[cp][0x300d,0x10a7].value # Récupère la valeur brute d'ouverture des lames
            tmp = tmp.decode('utf-8').strip('\x00').split('\\') # Décode et nettoie la chaîne de caractères
            sinogram[cp-1,:] = np.array(tmp,dtype=np.float64)   # Convertit en nombres et remplit la matrice
        except KeyError:
            continue                                         # Ignore s'il n'y a pas de données pour ce point
    return sinogram 
#===========================================================================================================

#======= Patient & Machine Info ============================================================================
def general_info(plan): 
    plan_info = {} 
    plan_info["patient_id"] = str(plan.PatientID)     # Extrait l'ID du patient
    plan_info["patient_name"] = str(plan.PatientName) # Extrait le nom brut du patient
    
    # --- recherche du nom du plan ---
    try:
        valeurs_trouvees = []
        # Cherche le nom du plan dans plusieurs balises DICOM (TPS potentiellement différents)
        if (0x0030, 0x0003) in plan and plan[0x0030, 0x0003].value:
            valeurs_trouvees.append(str(plan[0x0030, 0x0003].value))
        if (0x0030, 0x0002) in plan and plan[0x0030, 0x0002].value:
            valeurs_trouvees.append(str(plan[0x0030, 0x0002].value))
        if (0x0030, 0x0004) in plan and plan[0x0030, 0x0004].value:
            valeurs_trouvees.append(str(plan[0x0030, 0x0004].value))
        if (0x300a, 0x0004) in plan and plan[0x300a, 0x0004].value:
            valeurs_trouvees.append(str(plan[0x300a, 0x0004].value))

        # Supprime les textes identiques trouvés dans plusieurs balises
        valeurs_uniques = []
        for v in valeurs_trouvees:
            if v not in valeurs_uniques:
                valeurs_uniques.append(v)

        # Assemble les noms trouvés ou signale leur absence
        texte_brut = " | ".join(valeurs_uniques) if valeurs_uniques else "Aucun texte trouvé"
        plan_info["raw_site"] = texte_brut
        
    except Exception as e:
        plan_info["raw_site"] = "Erreur lecture"
        texte_brut = ""

   #===================AUTO-CATÉGORISATION================================================================
    texte_minuscule = texte_brut.lower() # Convertit tout en minuscules pour faciliter la recherche
    plan_info["treatment_site"] = "Inconnu" # Valeur par défaut
    
    for category, keywords in CATEGORIES_DICT.items():
        if any(mot in texte_minuscule for mot in keywords):
            plan_info["treatment_site"] = category
            break # On s'arrête de chercher dès qu'on a trouvé une correspondance
    #======================================================================================================

    #========= mapping ====================================================================================
    serial_mapping = {"4010012": "Tomo2", "210462": "Tomo4", "4010710": "Radi7"} 
    try:
        raw_serial = str(plan.DeviceSerialNumber) 
        plan_info["machine_nb"] = serial_mapping.get(raw_serial, raw_serial)
    except AttributeError:
        plan_info["machine_nb"] = "Unknown"
        
    return plan_info
    #===========================================================================================================

# region Traitement des fichiers DICOM 
def delivery_info(plan): 
    delivery = {}
    delivery["GP"] = float(plan.BeamSequence[0][0x300d,0x1040].value)  # Gantry Period (Temps de rotation)
    delivery["PT"] = (delivery["GP"]/51.0)*1000.0                      # Temps par projection (ms)
    delivery["CS"] = float(plan.BeamSequence[0][0x300d,0x1080].value)  # Vitesse de table (Couch Speed)
    delivery["pitch"] = float(plan.BeamSequence[0][0x300d,0x1060].value) # Facteur de pitch
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    delivery["Nrot"] = (NCP-1)/51                                      # Nombre de rotations totales
    delivery["TT"] = delivery["Nrot"]*delivery["GP"]                   # Temps total de traitement
    
    try:
        delivery["DS"] = float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose) # Dose par fraction
        delivery["Nb_Frac"] = int(plan.FractionGroupSequence[0].NumberOfFractionsPlanned)        # Nombre de séances
    except Exception:
        delivery["DS"] = 0.0
        delivery["Nb_Frac"] = 1
    return delivery
# endregion

#======= Error Calculation =================================================================================
def get_error_shift(sinogram, delivery):
    PT = delivery["PT"] 
    LOT_sino = PT*sinogram                    # Temps d'ouverture des lames par projection 
    maxLOT = np.max(LOT_sino)                 # Temps d'ouverture maximum
    total_lot = np.sum(LOT_sino)              # Temps d'ouverture cumulé sur tout le plan
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  # Filtre uniquement les lames qui s'ouvrent
    thresh = 18  
    undisc_LCT = 0                            # Compteur pour les erreurs LCT non discriminables
        
    error_per_projection = np.zeros(LOT_sino.shape[0]) # Stockage des erreurs par angle en MILLISECONDES
    
    # Conditions pour détecter un décalage potentiel des lames (Leaf Shift)
    cond1 = LOT_sino < (maxLOT-1)  
    cond2 = LOT_sino > (PT-thresh) 
    row, col = np.where(cond1 & cond2) 
                                            
    for i in range(len(row)):
        if row[i] < (LOT_sino.shape[0] - 1):             
            if (LOT_sino[row[i]+1, col[i]] > (PT-20)): 
                undisc_LCT += 1               # Incrémente si l'erreur impacte le temps de fermeture
        else: 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row>(-0.5)] 
    filtered_col = col[col>(-0.5)]
    extra_time = 0  
    
    # Calcul de la dose excessive théorique causée par la mécanique
    for i in range(len(filtered_row)-1):
        diff = (PT - LOT_sino[filtered_row[i], filtered_col[i]]) 
        extra_time += diff  
        error_per_projection[filtered_row[i]] += diff # Cumul des temps d'erreur en ms
    
    # Retourne l'erreur transfert (%), l'uLCT (%) et le profil d'erreur complet (en ms)
    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100), error_per_projection
#===========================================================================================================

#======= File Reading ======================================================================================
def read_files_in_directory(directory):
    files_found = []
    for root, dirs, files in os.walk(directory):      # Parcourt l'arborescence du dossier cible
        for file in files:
            if file.startswith(("RP", "RTPLAN")) and file.endswith(".dcm"):  # Filtre uniquement les plans de radiothérapie
                files_found.append(os.path.join(root, file))
    return files_found
#===========================================================================================================

#======= Dashboard Engine ==================================================================================
def display_dashboard(raw_data):
    if not raw_data: # Arrête l'affichage si aucune donnée n'est envoyée
        st.warning("Aucune donnée à afficher pour ce patient.")
        return

    # === Affichage de la Carte Patient ===
    patient_name = raw_data[0]['Patient']
    patient_id = raw_data[0]['ID']
    nb_frac_ref = raw_data[0]['Nb_Frac']
    site_traitement = raw_data[0].get('Site', 'Inconnu')  
    site_brut = raw_data[0].get('Raw_Site', 'Inconnu')    
    
    # Code HTML pour faire une jolie barre d'en-tête (Dossier, Séances, ID)
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

    # === Menu de correction manuelle ===
    with st.expander("Normaliser le nom de la localisation (Optionnel)"):
        st.markdown("<small style='color: #6c757d;'>Si le nom récupéré du DICOM comporte une faute de frappe ou est illisible, vous pouvez forcer un nom standard ici.</small>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        
        # Définition des catégories standards dynamiques (récupérées du fichier texte)
        liste_sites_propres = list(CATEGORIES_DICT.keys()) + ["Autre", "Inconnu"]

        # Définit l'index du menu déroulant pour qu'il affiche la valeur actuelle
        if site_traitement not in liste_sites_propres:
            options_affichage = [site_traitement] + liste_sites_propres
            default_idx = 0
        else:
            options_affichage = liste_sites_propres
            default_idx = liste_sites_propres.index(site_traitement)
            
        with col1: # Sélecteur Streamlit pour choisir la correction
            nouveau_site = st.selectbox("Sélectionnez la bonne localisation :", options_affichage, index=default_idx, key=f"select_site_{patient_id}")
            
        with col2: # Bouton d'action
            st.markdown("<br>", unsafe_allow_html=True) 
            if st.button("Valider la correction", type="primary", use_container_width=True, key=f"btn_site_{patient_id}"):
                
                # Mise à jour de la table SQL pour toutes les sessions de ce patient
                conn_update = sqlite3.connect(DB_NAME)
                cursor_update = conn_update.cursor()
                cursor_update.execute("""
                    UPDATE SESSIONS 
                    SET Treatment_Site = ? 
                    WHERE Patient_ID = ?
                """, (nouveau_site, patient_id))
                conn_update.commit()
                conn_update.close()
                
                # Feedback visuel avec notification Toast
                st.toast("Nom mis à jour avec succès !", icon="✅")
                time.sleep(0.5) 
                st.rerun() 
                
    st.markdown("<br>", unsafe_allow_html=True)

    # === Saisie des fractions (Validation par le clinicien) ===
    st.markdown("##### Validation des séances réalisées")
    st.markdown("<small style='color: #6c757d;'>Le DICOM ne reflétant pas toujours la réalité clinique de Mosaiq/Aria, merci de valider le nombre de séances réellement effectuées.</small>", unsafe_allow_html=True)
    
    fractions_done_list = []
    cols = st.columns(len(raw_data)) # Crée autant de colonnes qu'il y a de transferts
    
    for index in range(len(raw_data)):
        machine = raw_data[index]['Machine']
        if index < len(raw_data) - 1:
            # Calcule automatiquement les séances faites sur l'ancienne machine
            diff_auto = int(max(1, raw_data[index]['Nb_Frac'] - raw_data[index+1]['Nb_Frac']))
            with cols[index]:
                f_done = st.number_input(f"Faites sur {machine}", min_value=0, max_value=int(nb_frac_ref), value=diff_auto, key=f"frac_{index}")
            fractions_done_list.append(f_done)
        else:
            # Par défaut, 1 séance pour la machine actuelle
            with cols[index]:
                f_done = st.number_input(f"Aujourd'hui sur {machine}", min_value=0, max_value=int(nb_frac_ref), value=1, key=f"frac_{index}")
            fractions_done_list.append(f_done)
    st.markdown("<br>", unsafe_allow_html=True)

    # === Construction du tableau et calcul des doses cumulées ===
    # On récupère le seuil global en amont pour calculer le budget
    seuil_patient = st.session_state.get("seuil_global_val", 1.5)
    
    total_cumulated_dose = 0.0 
    final_table_data = [] 
    
    # NOUVEAU CALCUL DU BUDGET : Dose totale + % du seuil
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
        
        # Calcul de la dose réelle avec l'impact du transfert
        actual_dose_session = dose_nominal * (1 + (error_session_pct / 100.0)) 
        
        if index == 0:
            date_display = item['Date'] + " (Initiale)" 
            total_cumulated_dose += (dose_nominal * fractions_done) # Additionne la dose
            
            cumul_str = f"{total_cumulated_dose:.2f}"
            comment = f"Prescription : {dose_totale_prescrite:.2f} Gy"
            alert = False
            
            # Prépare la ligne d'historique pour le plan initial
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
            
            # Nouveau Système d'alerte : Uniquement basé sur le % d'erreur
            comments = []
            alert = False
            if error_session_pct > seuil_patient: 
                comments.append(f"Erreur > {seuil_patient}%")
                alert = True
                
            comment = " | ".join(comments) if comments else "OK"
        
            # Prépare la ligne d'historique pour un transfert
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
    
    # Fonction pour colorer le tableau (Initiale en bleu, alertes en rouge, code couleur par machine)
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

    # Affichage du tableau final
    styled_df = df.style.apply(style_dataframe, axis=1) 
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        height=200,
        column_config={ # Cache les colonnes techniques servant au calcul
            "_Alert": None,      
            "_Index": None,
            "_Cumul_Val": None,
            "_Budget_Total": None,
            "_Nb_Frac_Plan": None
        }
    )
    st.markdown("---")
    
    col_graph1, col_graph2 = st.columns([1, 1]) # Sépare la section en deux colonnes égales

    # --- Bloc de gauche : Simulation et Budget ---
    with col_graph1:
        st.subheader("🎯 Suivi du Budget Dose") 
        if len(df) > 0:
            last_session_df = df.iloc[-1]
            budget_max =  last_session_df['_Budget_Total']
            current_dose = last_session_df['_Cumul_Val']

            percentage = min(current_dose / budget_max, 1.0) if budget_max > 0 else 0.0
            
            # --- Nouvel affichage du budget mis en évidence ---
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
            # ---------------------------------------------------
            
            st.markdown("---")
            st.markdown("#### 🔮 Prévision de fin de traitement")

            # S'il y a eu un transfert, on simule la suite automatiquement avec la dernière machine
            if len(df) > 1:
                last_session_machine = df.iloc[-1]
                machine_actuelle = last_session_machine['Machine']
                
                if last_session_machine['Dose Délivrée (Gy/s.)'] == "-":
                    dose_par_seance_actuelle = float(last_session_machine['Dose Prévue (Gy/s.)'])
                else:
                    dose_par_seance_actuelle = float(last_session_machine['Dose Délivrée (Gy/s.)'])
                
                total_done_simul = sum(fractions_done_list)
                theoretical_remaining_sessions = nb_frac_ref - total_done_simul
                
                # Calcule le nombre max de séances faisables avant d'exploser le budget
                budget_remaining = budget_max - current_dose
                max_sessions_possibles = int(budget_remaining / dose_par_seance_actuelle) if dose_par_seance_actuelle > 0 and budget_remaining > 0 else 0
                
                # Calcule la dose que le patient prendra s'il fait toutes les séances restantes
                dose_finale_projetee = current_dose + (theoretical_remaining_sessions * dose_par_seance_actuelle)
                
                if theoretical_remaining_sessions > 0:
                    if max_sessions_possibles == theoretical_remaining_sessions:
                        alerte_bg = "#d4edda" 
                        alerte_text = "#155724"
                        alerte_message = f"✅ <b>Rythme Conforme :</b> Le patient peut faire ses <b>{theoretical_remaining_sessions} séances restantes</b> sur la {machine_actuelle} sans dépasser la limite de dose."
                    
                    elif max_sessions_possibles < theoretical_remaining_sessions:
                        alerte_bg = "#f8d7da" 
                        alerte_text = "#721c24"
                        perte = theoretical_remaining_sessions - max_sessions_possibles
                        alerte_message = f"⚠️ <b>ALERTE SURDOSE :</b> Au rythme de la {machine_actuelle} ({dose_par_seance_actuelle:.2f} Gy/s.), faire les {theoretical_remaining_sessions} séances prévues fera dépasser le budget total ({dose_finale_projetee:.2f} Gy).<br><br>👉 Il ne peut faire que <b>{max_sessions_possibles} séances supplémentaires maximum</b> (soit <b>-{perte} séance(s)</b> à annuler sur l'ordonnance)."
                    
                    else:
                        alerte_bg = "#fff3cd" 
                        alerte_text = "#856404"
                        gain = max_sessions_possibles - theoretical_remaining_sessions
                        alerte_message = f"ℹ️ <b>SOUS-DOSAGE :</b> La {machine_actuelle} délivrant moins que prévu, la limite ne sera pas atteinte à la fin du traitement. Il y a une marge pour rajouter <b>+{gain} séance(s)</b> si le médecin le juge nécessaire."

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

    # --- Bloc de droite : Graphique Polaire de l'Erreur ---
    with col_graph2:
        st.subheader("Localisation angulaire") 
        
        transfer_sessions = []
        # Identifie les réels transferts (changement de machine) pour les lister
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
                total_errors = np.array(session_to_analyze['Profile_Error']) # Tableau complet des erreurs en ms
                
                CS = session_to_analyze.get('CS', 0) 
                GP = session_to_analyze.get('GP', 0) 
                
                distance_tour_cm = (CS * GP) / 10.0 if CS and GP else 0  # Distance parcourue sur Gz par rotation
                n_rotations = len(total_errors) // 51                    # Nombre de rotations estimé
                
                # Curseur pour sélectionner la "tranche" à analyser si le plan a plusieurs rotations
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
                    slice_errors = total_errors[start:end] # Découpe le tableau pour n'afficher que 51 angles
                    caption_text = f"Temps d'erreur sur la rotation {rotation_target} (Gantry 0° - 360°)."
                else:
                    slice_errors = total_errors
                    caption_text = "Temps d'erreur sur 1 rotation (Gantry 0° - 360°)."
                
                # Configuration de Matplotlib pour tracer le cercle
                N_total = len(slice_errors)
                angles = np.linspace(0, 2 * np.pi, N_total, endpoint=False) 
                
                # Ferme la boucle graphique pour relier le dernier point au premier
                angles_closed = np.concatenate((angles, [angles[0]])) 
                errors_closed = np.concatenate((slice_errors, [slice_errors[0]])) 
                
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5, 5.5)) 
                ax.set_theta_zero_location("N") # Le 0° est en haut (Nord)
                ax.set_theta_direction(-1)      # Sens horaire
       
                # Ajustement de l'échelle du graphique (Max = erreur max)
                max_err_global = np.max(total_errors) 
                if max_err_global == 0:
                    max_err_global = 0.1
                ax.set_ylim(max_err_global * 1.3, 0) 

                ax.fill_between(angles_closed, 0, errors_closed, color='#ff4757', alpha=0.35) 
                ax.plot(angles_closed, errors_closed, color='#c0392b', linewidth=2.0, zorder=3)
                
                # Graduation angulaire (tous les 10 degrés)
                angles_deg = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                ax.set_xticks(angles_deg) 
                labels_10deg = [f"{i}°" for i in range(0, 360, 10)]
                ax.set_xticklabels(labels_10deg, fontsize=7, color='#2c3e50') 

                ax.set_facecolor('white') 
                tick_values = [max_err_global * 0.25, max_err_global * 0.5, max_err_global * 0.75, max_err_global]
                ax.set_yticks(tick_values) 
                
                # === MODIFICATION ICI : ms au lieu de % ===
                labels_ticks = [f"{val:.1f} ms" for val in tick_values]
                ax.set_yticklabels(labels_ticks, fontsize=7, color='#d32f2f', fontweight='bold') 
                
                ax.set_rlabel_position(25)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False) # Envoie le graphique à Streamlit
                st.caption(caption_text)
            else:
                st.info("No error detected in the selected session.")
#===========================================================================================================

#======= AUTOMATISATION DE L'INGESTION (Sans bouton) =======================================================
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Département de Physique Médicale**")
st.sidebar.markdown("---")

# ======= BOUTON PROCÉDURE (Téléchargement) =======
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
# =================================================

files_in = read_files_in_directory(DIR_IN)

# Si des fichiers sont présents, le traitement se lance tout seul !
if len(files_in) > 0:
    with st.sidebar.status(f"Intégration automatique de {len(files_in)} nouveau(x) plan(s)...", expanded=True) as status:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        duplicates_ignored = 0
        new_processed = 0

        # Boucle sur chaque fichier trouvé
        for f in files_in:
            try:
                # OPTIMISATION : Lecture unique du DICOM pour soulager le NAS
                plan = dcm.dcmread(f) 
                try:
                    uid = plan.SOPInstanceUID
                except AttributeError:
                    continue # Ignore si le fichier n'a pas d'UID
                
                try:
                    # Appel de toutes nos fonctions d'extraction
                    info = general_info(plan)
                    delivery = delivery_info(plan)
                    sinogram = get_sinogram(plan)
                    data = get_error_shift(sinogram, delivery) # data[0] = erreur transfert, data[1] = uLCT, data[2] = JSON
                    
                    # Formatage du nom "Nom^Prenom"
                    parts = info["patient_name"].split("^")  
                    full_name = f"{parts[1] if len(parts) > 1 else ''} {parts[0]}".strip()
                    
                    # Extraction de la date DICOM
                    try: plan_date = str(plan[0x0008, 0x0012].value) 
                    except KeyError: plan_date = "00000000" 
                    try: plan_time = str(plan[0x0008, 0x0013].value) 
                    except KeyError: plan_time = "000000"
                    
                    # Formate la date pour tri SQL et pour affichage humain
                    date_time_sort = f"{plan_date}{plan_time}"
                    display_date = f"{plan_date[6:8]}/{plan_date[4:6]}/{plan_date[0:4]} à {plan_time[0:2]}:{plan_time[2:4]}:{plan_time[4:6]}"
                    
                    # 1. Ajoute le patient dans la base (ignore s'il existe déjà)
                    cursor.execute("INSERT OR IGNORE INTO PATIENTS (Patient_ID, Full_Name) VALUES (?, ?)", 
                                   (info["patient_id"], full_name))
                                   
                    profile_json = json.dumps(data[2].tolist())
                    
                    # 2. Ajoute la session/transfert
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
                    duplicates_ignored += 1 # L'UID existait déjà, on évite le doublon
                    
            except Exception as e:
                st.sidebar.error(f"Erreur sur {os.path.basename(f)} : {e}")
            
            finally:
                # Archivage : Déplace le fichier traité pour vider le dossier de dépôt
                file_name = os.path.basename(f)
                dest_path = os.path.join(DIR_ARCHIVE, file_name)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(f, DIR_ARCHIVE)

                # Nettoyage des dossiers vides restants
                parent_dir = os.path.dirname(f)
                if parent_dir != DIR_IN and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
        
        conn.commit()
        conn.close()
        status.update(label="Traitement terminé !", state="complete", expanded=False)

    if new_processed > 0:
        # --- VIDAGE DU CACHE POUR METTRE À JOUR LES STATS ---
        charger_donnees_globales.clear()
        st.toast(f"{new_processed} plan(s) traité(s) avec succès !", icon="✅")
        time.sleep(1)
        st.rerun() # Relance l'application pour afficher les nouvelles données
else:
    st.sidebar.success(" Base de données à jour. En attente de nouveaux DICOM...")
#===========================================================================================================


#======= GESTION DE LA NAVIGATION ==========================================================================
st.markdown("<h1 style='text-align: center;'>Suivi de transfert TomoTherapy</h1>", unsafe_allow_html=True) 

# Initialisation de la mémoire de l'application
if "vue_actuelle" not in st.session_state:
    st.session_state.vue_actuelle = "Accueil" # Page par défaut
if "patient_cible" not in st.session_state:
    st.session_state.patient_cible = None

# ==========================================================================================================
# PAGE 1 : VUE GLOBALE (Accueil & Statistiques)
# ==========================================================================================================
if st.session_state.vue_actuelle == "Accueil":
    
    st.markdown("###  Contrôle Qualité Global du Service")
    
    # --- Curseur pour choisir le seuil dynamique ---
    seuil_alerte = st.number_input(
        "Définir le seuil d'alerte clinique par transfert (%) :", 
        min_value=0.0, max_value=10.0, value=1.5, step=0.1, key="seuil_global",
        help="Seuil de tolérance pour une séance individuelle. Définit le pourcentage d'erreur au-delà duquel la déviation mécanique lors d'un changement de machine est jugée critique."
    )
    # Sauvegarde la valeur dans le session_state pour qu'elle soit récupérable dans l'onglet Dossier
    st.session_state["seuil_global_val"] = seuil_alerte
    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Extraction globale via le Cache (Instantané)
    df_stats = charger_donnees_globales()

    if len(df_stats) == 0:
        st.info("Aucune donnée disponible pour le moment. Ingérez des fichiers DICOM pour générer les statistiques.")
    else:
        # Trie par Patient et par Date pour que les calculs de chronologie fonctionnent bien
        df_stats = df_stats.sort_values(by=['Patient_ID', 'Date_Time'])
        
        # --- PREPARATION DU TABLEAU RECAPITULATIF GLOBAL ---
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
            # Regroupe l'historique par ID Patient
            patients_dict = {}
            for row in toutes_sessions:
                pid = row[0]
                if pid not in patients_dict:
                    patients_dict[pid] = []
                patients_dict[pid].append(row)
            
            tableau_final = []
            
            # Analyse du dernier statut pour chaque patient
            for pid, sessions in patients_dict.items():
                plan_initial = sessions[0]
                plan_actuel = sessions[-1] # On regarde uniquement le dernier état connu
                
                nom_patient = plan_actuel[1]
                machine_actuelle = plan_actuel[2]
                dose_actuelle = plan_actuel[3]
                frac_restantes = plan_actuel[4]
                erreur_actuelle = plan_actuel[5]
                site = plan_actuel[6]
                date_display = plan_actuel[7]
                date_sort = plan_actuel[8] # Utilisé uniquement pour le tri caché
                
                if len(sessions) > 1: # Si le patient a subi au moins un transfert
                    # Calcul du nouveau budget autorisé en fonction du seuil en pourcentage
                    dose_init = plan_initial[3]
                    frac_init = plan_initial[4]
                    dose_totale_prescrite = dose_init * frac_init
                    budget_theorique = dose_totale_prescrite * (1 + (seuil_alerte / 100.0))
                    
                    # On ne teste plus que la condition d'alerte pour le % trop haut
                    alerte_erreur = erreur_actuelle > seuil_alerte
                    
                    if alerte_erreur:
                        statut = "🔴 % Trop haut"
                    else:
                        statut = "🟢 Conforme"
                    
                    erreur_str = f"{erreur_actuelle:.2f} %"
                else: # S'il n'a fait que son plan initial
                    statut = "⚪ Plan Initial"
                    erreur_str = "-"
                
                # Ajoute la ligne au tableau
                tableau_final.append({
                    "Date_Sort": date_sort, 
                    "Date (Dernier import)": date_display,
                    "ID Patient": pid,
                    "Nom": nom_patient,
                    "Localisation": site,
                    "Machine Destination": machine_actuelle,
                    "Erreur Transfert": erreur_str,
                    "Statut": statut
                })
            
            # Création du DataFrame et tri chronologique (les plus récents en haut)
            df_recap = pd.DataFrame(tableau_final)
            df_recap = df_recap.sort_values(by="Date_Sort", ascending=False).drop(columns=["Date_Sort"])
            
            # Fonction pour appliquer le code couleur de manière ciblée
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
                on_select="rerun",           # Recharge le script quand on clique
                selection_mode="single-row"  # On ne peut cliquer que sur un patient à la fois
            )
            
            # Détection du clic
            if len(event.selection.rows) > 0:
                index_clique = event.selection.rows[0]
                # Récupère l'ID du patient correspondant à la ligne cliquée
                patient_id_clique = df_recap.iloc[index_clique]["ID Patient"]
                
                # Mise à jour de la mémoire et changement de page
                st.session_state.patient_cible = patient_id_clique
                st.session_state.vue_actuelle = "Dossier"
                st.rerun() # Déclenche la bascule immédiate vers la page Dossier
        
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### Statistiques Globales")
        
        # --- CRÉATION DES SOUS-ONGLETS DE STATISTIQUES ---
        sub_tab_loc, sub_tab_mach, sub_tab_time, sub_tab_complex = st.tabs([
            " Par Localisation", 
            " Par Sens de Transfert", 
            " Évolution dans le Temps",
            " Complexité vs Erreur"
        ])

        # --------------------------------------------------------------------------------------------------
        # SOUS-ONGLET A : Localisation 
        # --------------------------------------------------------------------------------------------------
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

        # --------------------------------------------------------------------------------------------------
        # SOUS-ONGLET B : Sens de Transfert 
        # --------------------------------------------------------------------------------------------------
        with sub_tab_mach:
            st.markdown("<small style='color: #6c757d;'>Analyse par trajet de transfert et taux de dépassement du seuil clinique.</small><br>", unsafe_allow_html=True)

           #========== enleve la tomo initial=========================================
            df_mach = df_stats.copy()
            df_mach['Prev_Machine'] = df_mach.groupby('Patient_ID')['Machine'].shift(1)
            df_transitions = df_mach.dropna(subset=['Prev_Machine']).copy()
            df_transitions = df_transitions[df_transitions['Machine'] != df_transitions['Prev_Machine']]
            #=========================================================================

            #============== filtré les données ================================================================
            if len(df_transitions) == 0:
                st.info("Aucun changement inter-machines détecté dans la base pour le moment.")
            else:
                df_transitions['Trajet'] = df_transitions['Prev_Machine'] + " ➔ " + df_transitions['Machine']    # "Tomo4 ➔ Radi7"
                df_transitions['Depasse_Seuil'] = df_transitions['Error_pct'] > seuil_alerte # Identifie les plans qui dépassent le seuil dynamique
                
                # --- Filtre interactif ---
                liste_trajets = sorted(df_transitions['Trajet'].unique())
                trajet_choisi = st.selectbox(" Filtrer les compteurs et la répartition pour un trajet spécifique :", ["Tous les transferts (Global)"] + liste_trajets)
                if trajet_choisi == "Tous les transferts (Global)": # Filtrage dynamique selon le choix du menu déroulant
                    df_focus = df_transitions
                    titre_pie = "Répartition Globale"
                else:
                    df_focus = df_transitions[df_transitions['Trajet'] == trajet_choisi]
                    titre_pie = f"Répartition : {trajet_choisi}"
            #=================================================================================================

                #=====================Calcul des KPI sur la donnée filtrée=============================
                total_focus = len(df_focus)
                depassements_focus = df_focus['Depasse_Seuil'].sum()
                taux_focus = (depassements_focus / total_focus * 100) if total_focus > 0 else 0
                
                #Affichage des 3 KPI dynamiques
                col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                col_kpi1.metric(f"Total de Transferts", total_focus)
                col_kpi2.metric(f"Transferts > {seuil_alerte}% (Alerte)", int(depassements_focus))
                col_kpi3.metric(f"Taux d'alerte", f"{taux_focus:.1f} %")
                
                st.markdown("---")
                #================================================================================

                #====== appel des graphiques erreur moyenne et reparttion globale ==========================
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
                    
                    # La ligne rouge pointillée se déplace toute seule selon la valeur choisie !
                    fig_mach.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text=f"Seuil ({seuil_alerte}%)")
                    
                    st.plotly_chart(fig_mach, use_container_width=True)

                with col_chart2:
                    # Le Pie Chart s'adapte au filtre ET au seuil dynamique !
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

        # --------------------------------------------------------------------------------------------------
        # SOUS-ONGLET C : Évolution dans le Temps 
        # --------------------------------------------------------------------------------------------------
        with sub_tab_time:
            st.markdown("<small style='color: #6c757d;'>Suivi temporel de l'erreur moyenne pour détecter la fatigue des machines (maintenance prédictive). <b>Les plans initiaux sont exclus.</b></small><br><br>", unsafe_allow_html=True)
            
            df_time = df_stats.copy()

            # --- EXCLUSION DES PLANS INITIAUX ---
            df_time = df_time.sort_values(by=['Patient_ID', 'Date_Time'])
            df_time['session_num'] = df_time.groupby('Patient_ID').cumcount()
            df_time = df_time[df_time['session_num'] > 0] # On ne garde que les transferts !
            # ------------------------------------------------
            
            # CORRECTION : Le format DICOM rajoute souvent des millisecondes (ex: 20260512143022.123).
            df_time['Date_Clean'] = df_time['Date_Time'].astype(str).str[:14]
            df_time['True_Date'] = pd.to_datetime(df_time['Date_Clean'], format='%Y%m%d%H%M%S', errors='coerce')
            
            # Nettoie les lignes où la date DICOM était illisible
            df_time = df_time.dropna(subset=['True_Date'])
            
            if len(df_time) == 0:
                st.warning("Pas de dates valides trouvées pour tracer l'évolution.")
            else:
                # Choix dynamique de la précision du temps
                granularite = st.radio(" Précision de la chronologie :", options=["Moyenne par Jour", "Moyenne par Semaine", "Moyenne par Mois"], horizontal=True)
                
                # Traduit le choix en code pour Pandas (D = Jour, W = Semaine, M = Mois)
                if granularite == "Moyenne par Jour":
                    freq = "D"
                elif granularite == "Moyenne par Semaine":
                    freq = "W"
                else:
                    freq = "M"

                # Regroupe les dates selon le choix
                df_time['Periode'] = df_time['True_Date'].dt.to_period(freq).dt.to_timestamp()
                
                # Calcule la moyenne d'erreur par période et par machine
                df_trend = df_time.groupby(['Periode', 'Machine'])['Error_pct'].mean().reset_index()
                
                # === Ajout des KPIs de dérive ============================
                st.markdown("#### Evolution de l'erreur (Dernière période vs Précédente)")
                
                # Trie les machines par ordre alphabétique pour figer l'affichage (ex: Radi7, Tomo2, Tomo4)
                machines_presentes = sorted(df_trend['Machine'].unique())
                
                if len(machines_presentes) > 0:
                    colonnes_kpi = st.columns(len(machines_presentes))
                    
                    for i, mach in enumerate(machines_presentes):
                        # Isole les données de la machine et les trie par ordre chronologique
                        df_mach = df_trend[df_trend['Machine'] == mach].sort_values(by='Periode')
                        
                        if len(df_mach) >= 2:
                            current_err = df_mach.iloc[-1]['Error_pct']  # Dernière période
                            previous_err = df_mach.iloc[-2]['Error_pct'] # Période juste avant
                            delta_err = current_err - previous_err
                            
                            colonnes_kpi[i].metric(
                                label=f"Tendance {mach}",
                                value=f"{current_err:.2f} %",
                                delta=f"{delta_err:+.2f} %",
                                delta_color="inverse" # Inverse : hausse (positif) = rouge = mauvais
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
                # ==============================================================================

                # Trace le graphique en ligne
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

        # --------------------------------------------------------------------------------------------------
        # SOUS-ONGLET D : Complexité vs Erreur (Analyse Physique)
        # --------------------------------------------------------------------------------------------------
        with sub_tab_complex:
            st.markdown("<small style='color: #6c757d;'>Corrélation entre les paramètres cinématiques du plan (Vitesse de rotation, Vitesse de table) et les erreurs de transfert. <b>Les plans initiaux sont exclus.</b></small><br><br>", unsafe_allow_html=True)
                    
            # Choix interactif pour l'utilisateur
            param_choice = st.radio("Sélectionnez le paramètre physique en abscisse (X) :", options=["Gantry Period (Temps de rotation en s)", "Couch Speed (Vitesse de table en mm/s)"], horizontal=True)
                    
            # Adapte la colonne de la base de données en fonction du choix
            x_col = 'GP_s' if 'Gantry' in param_choice else 'CS_mm_s'
            x_label = 'Gantry Period (s)' if 'Gantry' in param_choice else 'Couch Speed (mm/s)'
                    
            # Nettoie les données (enlève les plans où le paramètre n'a pas été lu correctement)
            df_complex = df_stats.dropna(subset=[x_col, 'Error_pct']).copy()
                    
            # --- CORRECTION : EXCLUSION DES PLANS INITIAUX ---
            df_complex['session_num'] = df_complex.groupby('Patient_ID').cumcount()
            df_complex = df_complex[df_complex['session_num'] > 0]
            # -------------------------------------------------
                    
            # Sécurité supplémentaire pour éviter les valeurs aberrantes à 0 ou négatives
            df_complex = df_complex[df_complex[x_col] > 0]
                    
            if len(df_complex) == 0:
                st.warning("Aucun transfert avec des données physiques valides n'est disponible pour générer le graphique.")
            else:
                # === ÉTAPE CRUCIALE : Tri des valeurs par l'axe X pour lier les points proprement de gauche à droite ===
                df_complex = df_complex.sort_values(by=x_col)
                
                # Création du graphique en ligne avec marqueurs (remplace px.scatter)
                fig_complex = px.line(
                    df_complex,
                    x=x_col,
                    y='Error_pct',
                    color='Machine',            # Uniquement trié et coloré par machine
                    markers=True,               # Affiche les points ET les relie par une ligne
                    hover_data=['Patient_ID', 'Raw_Treatment_Site'], 
                    title=f"Évolution de l'Erreur en fonction du {x_label} par Machine",
                    labels={x_col: x_label, 'Error_pct': 'Erreur Moyenne (%)', 'Machine': 'Machine Tomo'},
                )
                        
                # Mise en page
                fig_complex.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40, b=50, l=0, r=0))
                fig_complex.add_hline(y=seuil_alerte, line_dash="dash", line_color="red", annotation_text=f"Seuil d'alerte ({seuil_alerte}%)")
                        
                # === Indicateurs de vitesse sous l'axe X ===
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

# ==========================================================================================================
# PAGE 2 : DOSSIER PATIENT 
# ==========================================================================================================
elif st.session_state.vue_actuelle == "Dossier":
    
    # Bouton pour revenir à la vue globale
    if st.button("⬅️ Retour au tableau de bord général"):
        st.session_state.vue_actuelle = "Accueil"
        st.session_state.patient_cible = None
        st.rerun()

    st.markdown("---")
    
    id_target = st.session_state.patient_cible
    
    with st.spinner("Récupération du dossier..."): 
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.Display_Date, s.Machine, s.Dose_Gy, s.Nb_Frac, s.uLCT, s.Error_pct, s.Profile_JSON, s.Date_Time, p.Full_Name, s.CS_mm_s, s.GP_s, s.Treatment_Site, s.Raw_Treatment_Site 
            FROM SESSIONS s
            JOIN PATIENTS p ON s.Patient_ID = p.Patient_ID
            WHERE s.Patient_ID = ? 
            ORDER BY s.Date_Time ASC
        """, (id_target,))
        
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