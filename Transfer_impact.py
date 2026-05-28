#======= Import bibliothèques ===============================================================================
import streamlit as st
import pydicom as dcm
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import time
import sqlite3
import json
#===========================================================================================================

#======= Configuration de la page web ======================================================================
st.set_page_config(page_title="Transfert Tomo", layout="wide")
#===========================================================================================================

#======= Création automatique des dossiers & Base de données ===============================================
DIR_IN = "IN"
DIR_ARCHIVE = "ARCHIVES"
DB_NAME = "tomo_database.db"

os.makedirs(DIR_IN, exist_ok=True) 
os.makedirs(DIR_ARCHIVE, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS PATIENTS (
        Patient_ID TEXT PRIMARY KEY,
        Nom_Complet TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS SEANCES (
        Seance_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Patient_ID TEXT,
        DICOM_SOP_UID TEXT UNIQUE,
        Date_Heure TEXT,
        Date_Affichee TEXT,
        Machine TEXT,
        Dose_Gy REAL,
        Nb_Frac INTEGER,
        uLCT REAL,
        Erreur_pct REAL,
        Profil_JSON TEXT,
        FOREIGN KEY (Patient_ID) REFERENCES PATIENTS(Patient_ID)
    )
    ''')
    conn.commit()
    conn.close()

init_db()
#===========================================================================================================

#=======  Récupération sinogrammes    ======================================================================
def get_sinogram(plan):
    NCP = plan.BeamSequence[0].NumberOfControlPoints
    sinogram = np.zeros((NCP,64)) 
    cp_sequence = plan.BeamSequence[0].ControlPointSequence 

    for cp in range(NCP): 
        try : 
            tmp = cp_sequence[cp][0x300d,0x10a7].value 
            tmp = tmp.decode('utf-8').strip('\x00').split('\\') 
            sinogram[cp-1,:] = np.array(tmp,dtype=np.float64) 
        except KeyError:
            continue 
    return sinogram 
#===========================================================================================================

#=======  Récupération Nom + Prénom + Machine  =============================================================
def general_info(plan): 
    plan_info = {} 
    plan_info["patient_id"] = str(plan.PatientID)
    plan_info["patient_name"] = str(plan.PatientName)
    serial_mapping = {"4010012": "Tomo2", "210462": "Tomo4", "4010710": "Tomo7"} 
    
    try:
        raw_serial = str(plan.DeviceSerialNumber) 
        plan_info["machine_nb"] = serial_mapping.get(raw_serial, raw_serial)
    except AttributeError:
        plan_info["machine_nb"] = "Inconnue"
    return plan_info
#===========================================================================================================

#=======  Récupération paramètres physiques  ===============================================================
def delivery_info(plan): 
    delivery = {}
    delivery["GP"] = float(plan.BeamSequence[0][0x300d,0x1040].value) 
    delivery["PT"] = (delivery["GP"]/51.0)*1000.0   
    delivery["CS"] = float(plan.BeamSequence[0][0x300d,0x1080].value) 
    delivery["pitch"] = float(plan.BeamSequence[0][0x300d,0x1060].value) 
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    delivery["Nrot"] = (NCP-1)/51   
    delivery["TT"] = delivery["Nrot"]*delivery["GP"] 
    
    try:
        delivery["DS"] = float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose) 
        delivery["Nb_Frac"] = int(plan.FractionGroupSequence[0].NumberOfFractionsPlanned) 
    except Exception:
        delivery["DS"] = 0.0
        delivery["Nb_Frac"] = 1
    return delivery
#===========================================================================================================

#=======  Calcul des erreurs    ============================================================================
def get_error_shift(sinogram,delivery):
    PT = delivery["PT"] 
    LOT_sino = PT*sinogram  
    maxLOT = np.max(LOT_sino) 
    total_lot = np.sum(LOT_sino) 
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  
    thresh = 18  
    undisc_LCT = 0
        
    erreur_par_projection = np.zeros(LOT_sino.shape[0]) 
    
    cond1 = LOT_sino < (maxLOT-1)  
    cond2 = LOT_sino > (PT-thresh) 
    row,col = np.where(cond1 & cond2) 
                                      
    for i in range(len(row)):
        if row[i] < (LOT_sino.shape[0] - 1):             
            if (LOT_sino[row[i]+1,col[i]] > (PT-20)): 
                undisc_LCT += 1 
        else: 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row>(-0.5)] 
    filtered_col = col[col>(-0.5)]
    extra_time = 0  
    
    for i in range(len(filtered_row)-1):
        diff = (PT - LOT_sino[filtered_row[i],filtered_col[i]])
        extra_time += diff  
        erreur_par_projection[filtered_row[i]] += diff 
    
    erreur_par_projection_pct = (erreur_par_projection / total_lot) * 100 
    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100), erreur_par_projection_pct
#===========================================================================================================

#=======  Lire les fichiers    =============================================================================
def lire_fichiers_dossier(dossier):
    fichiers_trouves = []
    for root, dirs, files in os.walk(dossier):  
        for file in files:
            # On accepte désormais les fichiers commençant par "RP" OU "RTPLAN"
            if file.startswith(("RP", "RTPLAN")) and file.endswith(".dcm"):  
                fichiers_trouves.append(os.path.join(root, file))
    return fichiers_trouves
#===========================================================================================================

#=======  Moteur d'affichage (Le "Peintre") ================================================================
def afficher_dashboard(raw_data):
    if not raw_data:
        st.warning("Aucune donnée à afficher pour ce patient.")
        return

    dose_cumulee_totale = 0.0
    final_table_data = []
    
    try:
        dose_nominale_ref = raw_data[0]['Dose (Gy)']
        nb_frac_ref = raw_data[0]['Nb_Frac']
        budget_total_Gy = (dose_nominale_ref * nb_frac_ref) + 0.5 
    except:
        budget_total_Gy = 0.0

    for index, item in enumerate(raw_data):
        error_seance_pct = item['Erreur Séance (%)']
        dose_nominale = item['Dose (Gy)']
        
        dose_reelle_seance = dose_nominale * (1 + (error_seance_pct / 100.0))
        
        
        if index == 0:
            date_affiche = item['Date'] + " (Initiale)"
            dose_cumulee_totale += dose_nominale 
            budget_restant = budget_total_Gy - dose_cumulee_totale
            seances_max_possibles = int(budget_restant / dose_nominale) if dose_nominale > 0 else 0
            
            
            seances_totales = 1 + seances_max_possibles 
            
            cumul_str = f"{dose_cumulee_totale:.2f}"
            commentaire = f"Budget prescript : {budget_total_Gy:.2f} Gy"
            alerte = False
            running_cumul_val = dose_cumulee_totale
            
            final_table_data.append({
                "Date": date_affiche, "Machine": item['Machine'],
                "Dose Prévue (Gy)": f"{dose_nominale:.2f}", 
                "uLCT (%)": f"{item['uLCT (%)']:.2f}",
                "Erreur Séance (%)": "-",
                "Dose Délivrée (Gy)": "-", 
                "Dose Cumulée (Gy)": cumul_str, 
                "Séances Prévue ": str(seances_totales), 
                "Commentaire": commentaire, 
                "_Alerte": alerte, "_Index": index, "_Cumul_Val": running_cumul_val, "_Budget_Total": budget_total_Gy
            })
        else:
            dose_cumulee_totale += dose_reelle_seance
            budget_restant = budget_total_Gy - dose_cumulee_totale
            
            if dose_reelle_seance > 0 and budget_restant > 0:
                seances_max_possibles = max(0, int(budget_restant / dose_reelle_seance))
            else:
                seances_max_possibles = 0
                
            
            seances_totales = (index + 1) + seances_max_possibles
                
            running_cumul_val = dose_cumulee_totale
            cumul_str = f"{dose_cumulee_totale:.2f}"
            
            commentaires = []
            alerte = False
            if error_seance_pct > 1.5: commentaires.append("Erreur > 1.5%"); alerte = True
            if dose_cumulee_totale >= budget_total_Gy: commentaires.append("BUDGET DÉPASSÉ !"); alerte = True
            commentaire = " | ".join(commentaires) if commentaires else "OK"
        
            final_table_data.append({
                "Date": item['Date'], "Machine": item['Machine'],
                "Dose Prévue (Gy)": f"{dose_nominale:.2f}", 
                "uLCT (%)": f"{item['uLCT (%)']:.2f}",
                "Erreur Séance (%)": f"{error_seance_pct:.2f}",
                "Dose Délivrée (Gy)": f"{dose_reelle_seance:.2f}",
                "Dose Cumulée (Gy)": cumul_str, 
                "Séances Totales (Faites + Restantes)": str(seances_totales), 
                "Commentaire": commentaire, 
                "_Alerte": alerte, "_Index": index, "_Cumul_Val": running_cumul_val, "_Budget_Total": budget_total_Gy
            })


    # Affichage du header patient
    c1, c2 = st.columns(2) 
    c1.info(f"**Patient :** {raw_data[0]['Patient']}")
    c2.info(f"**ID :** {raw_data[0]['ID']}")

    df = pd.DataFrame(final_table_data) 
    
    def style_dataframe(row):
        styles = []
        ligne_style = ''
        if row['_Index'] == 0: 
            ligne_style = 'background-color: #f0f8ff; font-weight: bold;' 
        elif row['_Alerte']: 
            ligne_style = 'background-color: #ffebee; color: #d32f2f; font-weight: bold;' 
            
        for col in row.index:
            cell_style = ligne_style 
            if col == 'Machine':
                if row['Machine'] == 'Tomo2': cell_style = 'background-color: #bbdefb; color: #000000; font-weight: bold;' 
                elif row['Machine'] == 'Tomo4': cell_style = 'background-color: #ffcdd2; color: #000000; font-weight: bold;' 
                elif row['Machine'] == 'Tomo7': cell_style = 'background-color: #c8e6c9; color: #000000; font-weight: bold;' 
            styles.append(cell_style)
        return styles

    styled_df = df.style.apply(style_dataframe, axis=1) 
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        height=200,
        column_config={
            "_Alerte": None,      
            "_Index": None,
            "_Cumul_Val": None,
            "_Budget_Total": None 
        }
    )
    st.markdown("---")
    
    col_graph1, col_graph2 = st.columns([1, 1])

    with col_graph1:
        st.subheader("Consommation du Budget Dose") 
        if len(df) > 0:
            derniere_seance_df = df.iloc[-1]
            budget_max = derniere_seance_df['_Budget_Total']
            dose_actuelle = derniere_seance_df['_Cumul_Val']

            pourcentage = min(dose_actuelle / budget_max, 1.0) if budget_max > 0 else 0.0
            
            st.progress(pourcentage)
            st.markdown(f"<h3 style='text-align: center; color: #333;'>{dose_actuelle:.2f} Gy / {budget_max:.2f} Gy</h3>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### Simulateur de fin de traitement")

            if len(df) > 1: 
                machines_uniques = df['Machine'].unique().tolist()
                machine_choisie = st.radio("Projeter la suite du traitement avec :", options=machines_uniques, horizontal=True)
                
                derniere_seance_machine = df[df['Machine'] == machine_choisie].iloc[-1]
                
                if derniere_seance_machine['Dose Délivrée (Gy)'] == "-":
                    dose_simulee = float(derniere_seance_machine['Dose Prévue (Gy)'])
                else:
                    dose_simulee = float(derniere_seance_machine['Dose Délivrée (Gy)'])
                
                budget_restant = budget_max - dose_actuelle
                seances_max = int(budget_restant / dose_simulee) if dose_simulee > 0 and budget_restant > 0 else 0
                
                seances_faites = len(df)
                nb_frac_ref = raw_data[0]['Nb_Frac']
                seances_restantes_theoriques = nb_frac_ref - seances_faites
                
                if seances_max > 0:
                    perte = seances_restantes_theoriques - seances_max
                    alerte_perte = f" <i>(soit une perte de <b>{perte} séance(s)</b>)</i>" if perte > 0 else ""
                    
                    couleur_fond = "#f0f8ff" 
                    if machine_choisie == 'Tomo2': couleur_fond = "#bbdefb" 
                    elif machine_choisie == 'Tomo4': couleur_fond = "#ffcdd2" 
                    elif machine_choisie == 'Tomo7': couleur_fond = "#c8e6c9" 
                        
                    html_texte = f"""
                    <div style="background-color: {couleur_fond}; padding: 16px; border-radius: 8px; color: #000000; margin-top: 10px;">
                        <b>Si le patient continue sur {machine_choisie} :</b><br><br>
                        Il a déjà réalisé <b>{seances_faites} séance(s)</b>.<br><br>
                        Au rythme de cette machine (<b>{dose_simulee:.2f} Gy</b>), il peut encore faire <b>{seances_max} séances</b> au lieu des <b>{seances_restantes_theoriques}</b> initialement prévues{alerte_perte}.
                    </div>
                    """
                    st.markdown(html_texte, unsafe_allow_html=True)
                else:
                    st.error(f" **ALERTE CRITIQUE :** Le budget total sera dépassé à la prochaine séance sur la {machine_choisie} !")
            else:
                dose_derniere = float(derniere_seance_df['Dose Prévue (Gy)'])
                nb_frac_ref = raw_data[0]['Nb_Frac']
                st.info(f" **Plan initial :** Rythme nominal de {dose_derniere:.2f} Gy/séance. Le patient doit faire **{nb_frac_ref} séances** au total.")

    with col_graph2:
        st.subheader("Localisation angulaire (Tomo-vue)")
        derniere_seance = raw_data[-1] 

        if derniere_seance['Erreur Séance (%)'] > 0.0:
            erreurs = np.array(derniere_seance['Profil_Erreur'])
            N_total = len(erreurs) 
            angles = np.linspace(0, 2 * np.pi, N_total, endpoint=False) 
            
            angles_fermes = np.concatenate((angles, [angles[0]])) 
            erreurs_fermees = np.concatenate((erreurs, [erreurs[0]])) 
            
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5, 5.5)) 
            ax.set_theta_zero_location("N") 
            ax.set_theta_direction(-1) 
            
            max_err = np.max(erreurs) 
            ax.set_ylim(max_err * 1.4, 0) 

            ax.fill_between(angles_fermes, 0, erreurs_fermees, color='#FF4B4B', alpha=0.7) 
            ax.plot(angles_fermes, erreurs_fermees, color='red', linewidth=1.5) 
            
            angles_51 = np.linspace(0, 2 * np.pi, 51, endpoint=False)
            ax.set_xticks(angles_51) 
            ax.set_xticklabels([str(i+1) for i in range(51)], fontsize=6) 

            ax.set_facecolor('white') 
            ax.set_yticks([max_err * 0.25, max_err * 0.5, max_err * 0.75, max_err]) 
            ax.set_yticklabels([]) 
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)
            st.caption("Excès de dose ramené sur 1 rotation (51 projections).")
        else:
            st.success("Aucune erreur détectée sur la dernière séance.")
#===========================================================================================================

#=======  BARRE LATÉRALE (Menu Ingestion) ==================================================================
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Service de Physique Médicale**")
st.sidebar.markdown("---")
st.sidebar.markdown("###  Ingestion des plans")
st.sidebar.markdown(f"<small>Traitez les fichiers DICOM déposés dans le dossier **Transfert_tomo** pour les intégrer à la base.</small>", unsafe_allow_html=True)

fichiers_in = lire_fichiers_dossier(DIR_IN)

if st.sidebar.button("Traiter les nouveaux plans", use_container_width=True, type="primary"):
    if len(fichiers_in) == 0:
        st.sidebar.warning(f"Aucun nouveau fichier trouvé.")
    else:
        with st.sidebar.status("Analyse en cours...", expanded=True) as status:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            doublons_ignores = 0
            nouveaux_traites = 0

            for f in fichiers_in:
                try:
                    ds = dcm.dcmread(f, stop_before_pixels=True)
                    try:
                        uid = ds.SOPInstanceUID
                    except AttributeError:
                        continue
                    
                    try:
                        plan = dcm.dcmread(f)
                        info = general_info(plan)
                        delivery = delivery_info(plan)
                        sinogram = get_sinogram(plan)
                        data = get_error_shift(sinogram, delivery)
                        
                        parts = info["patient_name"].split("^")  
                        nom_complet = f"{parts[1] if len(parts) > 1 else ''} {parts[0]}".strip()
                        
                        try: plan_date = str(plan[0x0008, 0x0012].value) 
                        except KeyError: plan_date = "00000000" 
                        try: plan_time = str(plan[0x0008, 0x0013].value) 
                        except KeyError: plan_time = "000000"
                        
                        date_heure_tri = f"{plan_date}{plan_time}"
                        date_affichee = f"{plan_date[6:8]}/{plan_date[4:6]}/{plan_date[0:4]} à {plan_time[0:2]}:{plan_time[2:4]}:{plan_time[4:6]}"
                        
                        cursor.execute("INSERT OR IGNORE INTO PATIENTS (Patient_ID, Nom_Complet) VALUES (?, ?)", 
                                       (info["patient_id"], nom_complet))
                                       
                        profil_json = json.dumps(data[2].tolist())
                        
                        cursor.execute("""
                            INSERT INTO SEANCES (
                                Patient_ID, DICOM_SOP_UID, Date_Heure, Date_Affichee, Machine, 
                                Dose_Gy, Nb_Frac, uLCT, Erreur_pct, Profil_JSON
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (info["patient_id"], uid, date_heure_tri, date_affichee, info["machine_nb"], 
                              delivery["DS"], delivery.get("Nb_Frac", 1), data[1], data[0], profil_json))
                        
                        nouveaux_traites += 1
                        
                    except sqlite3.IntegrityError:
                        doublons_ignores += 1
                        
                except Exception as e:
                    st.sidebar.error(f"Erreur sur {os.path.basename(f)} : {e}")
                
                nom_fichier = os.path.basename(f)
                chemin_dest = os.path.join(DIR_ARCHIVE, nom_fichier)
                if os.path.exists(chemin_dest):
                    os.remove(chemin_dest)
                shutil.move(f, DIR_ARCHIVE)
            
            conn.commit()
            conn.close()
            status.update(label="Traitement terminé !", state="complete", expanded=False)

        if nouveaux_traites > 0:
            st.sidebar.success(f"**{nouveaux_traites}** nouveau(x) plan(s) ajouté(s).")
        else:
            st.sidebar.info(f"Aucun ajout. **{doublons_ignores}** doublon(s) archivé(s).")
#===========================================================================================================

#=======  ÉCRAN PRINCIPAL (Analyse Clinique) ===============================================================
st.markdown("<h1 style='text-align: center;'>Suivi des doses Tomo</h1>", unsafe_allow_html=True) 

st.markdown("### Rechercher l'historique d'un patient")

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT Patient_ID, Nom_Complet FROM PATIENTS")
patients_db = cursor.fetchall()

if len(patients_db) == 0:
    st.info(f"La base de données est vide. Déposez des fichiers dans le dossier **{DIR_IN}** et cliquez sur le bouton à gauche.")
else:
    liste_choix = sorted([f"{p[1]} (ID: {p[0]})" for p in patients_db])

    new_icon_url = "https://img.icons8.com/?size=25&id=7eX13e1GI7bn&format=png&color=000000"
    st.markdown(f' <img src="{new_icon_url}" style="height: 20px; vertical-align: middle;"> Rechercher par Nom, Prénom ou ID :', unsafe_allow_html=True)
    recherche = st.text_input("", placeholder="Ex: Dupont, Jean, ou 12345...")

    if recherche:
        liste_choix = [p for p in liste_choix if recherche.lower() in p.lower()] 
                
    if len(liste_choix) == 0: 
        st.warning("Aucun patient ne correspond à cette recherche.")
    else:
        patient_selectionne = st.selectbox("Sélectionnez un patient :", ["-- Choisir un patient --"] + liste_choix) 
        
        if patient_selectionne != "-- Choisir un patient --":
            id_cible = patient_selectionne.split("ID: ")[1].replace(")", "")
            
            with st.spinner("Récupération rapide depuis la base SQL..."): 
                cursor.execute("""
                    SELECT s.Date_Affichee, s.Machine, s.Dose_Gy, s.Nb_Frac, s.uLCT, s.Erreur_pct, s.Profil_JSON, s.Date_Heure, p.Nom_Complet 
                    FROM SEANCES s
                    JOIN PATIENTS p ON s.Patient_ID = p.Patient_ID
                    WHERE s.Patient_ID = ? 
                    ORDER BY s.Date_Heure ASC
                """, (id_cible,))
                
                lignes_seances = cursor.fetchall()
                
                raw_data_sql = []
                for row in lignes_seances:
                    raw_data_sql.append({
                        'Date': row[0],
                        'Machine': row[1],
                        'Dose (Gy)': row[2],
                        'Nb_Frac': row[3],
                        'uLCT (%)': row[4],
                        'Erreur Séance (%)': row[5],
                        'Profil_Erreur': json.loads(row[6]),
                        'sort_key': row[7],
                        'Patient': row[8],
                        'ID': id_cible
                    })
                    
                afficher_dashboard(raw_data_sql)

