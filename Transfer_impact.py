import streamlit as st
import pydicom as dcm
import numpy as np
import pandas as pd
import os
import shutil

# Configuration de la page web
st.set_page_config(page_title="transfert tomo", layout="wide")

st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("---") # Ajoute une petite ligne de séparation en dessous
st.sidebar.markdown("**Service de Physique Médicale**")

# Création automatique des dossiers IN et OUT
DIR_IN = "IN"
DIR_OUT = "OUT"
os.makedirs(DIR_IN, exist_ok=True)
os.makedirs(DIR_OUT, exist_ok=True)


def get_sinogram(plan):
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    sinogram = np.zeros((NCP,64)) 
    cp_sequence = plan.BeamSequence[0].ControlPointSequence 
    for cp in range(NCP): 
        try : 
            tmp = cp_sequence[cp][0x300d,0x10a7].value 
            tmp = tmp.decode('utf-8').split('\\')
            sinogram[cp-1,:] = np.array(tmp,dtype=np.float64)  
        except KeyError:
            continue 
    return sinogram 

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
    except Exception:
        delivery["DS"] = 0.0
    return delivery

def get_error_shift(sinogram,delivery):
    PT = delivery["PT"] 
    LOT_sino = PT*sinogram
    maxLOT = np.max(LOT_sino)
    total_lot = np.sum(LOT_sino)
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  
    thresh = 18 
    undisc_LCT = 0
    
    cond1 = LOT_sino < (maxLOT-1)  
    cond2 = LOT_sino > (PT-thresh) 
    row,col = np.where(cond1 & cond2)
    
    for i in range(len(row)):
        if row[i] < LOT_sino.shape[0]:            
            if (LOT_sino[row[i]+1,col[i]] > (PT-20)):
                undisc_LCT += 1 
        else: 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row>(-0.5)] 
    filtered_col = col[col>(-0.5)]
    extra_time = 0  
    for i in range(len(filtered_row)-1):
        extra_time += (PT - LOT_sino[filtered_row[i],filtered_col[i]])  

    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100) 

def lire_fichiers_dossier(dossier):
    fichiers_trouves = []
    for root, dirs, files in os.walk(dossier):  
        for file in files:
            if file.startswith("RP") and file.endswith(".dcm"):  
                fichiers_trouves.append(os.path.join(root, file))
    return fichiers_trouves


def analyser_et_afficher_tableau(fichiers_a_traiter):
    raw_data = []
    plans_vus = set()
    doublons_ignores = 0
    
    # 1. Extraction des données
    for chemin_fichier in fichiers_a_traiter:
        plan = dcm.dcmread(chemin_fichier)
        info = general_info(plan)
        
        parts = info["patient_name"].split("^")  
        name_id = f"{parts[1] if len(parts) > 1 else ''} {parts[0]}".strip()
        
        try: plan_date = str(plan[0x0008, 0x0012].value)
        except KeyError: plan_date = "00000000" 
        try: plan_time = str(plan[0x0008, 0x0013].value)
        except KeyError: plan_time = "000000"
        
        cle_unique = f"{info['patient_id']}_{plan_date}_{plan_time}"
        
        if cle_unique in plans_vus:
            doublons_ignores += 1
            continue
        
        plans_vus.add(cle_unique)
        
        sinogram = get_sinogram(plan) 
        delivery = delivery_info(plan) 
        data = get_error_shift(sinogram, delivery) 

        raw_data.append({
            'sort_key': plan_date + plan_time, 
            'Date': f"{plan_date[6:8]}/{plan_date[4:6]}/{plan_date[0:4]} à {plan_time[0:2]}:{plan_time[2:4]}:{plan_time[4:6]}",
            'Patient': name_id,
            'ID': str(info["patient_id"]),
            'Machine': info["machine_nb"],
            'Dose (Gy)': delivery["DS"],
            'uLCT (%)': data[1],
            'Erreur Séance (%)': data[0]
        })

    if not raw_data:
        return doublons_ignores, False

    # 2. Tri et calculs cumulatifs
    raw_data.sort(key=lambda x: x['sort_key'])
    
    total_cumulative_error = 0.0
    for index, item in enumerate(raw_data):
        if index > 0: 
            total_cumulative_error += item['Erreur Séance (%)']

    running_cumulative_error = 0.0
    final_table_data = []

    for index, item in enumerate(raw_data):
        error_seance = item['Erreur Séance (%)']
        
        if index == 0:
            item['Date'] += " (Initiale)"
            cumul_str = "-"
            seances_str = "-"
            commentaire = "Plan de référence"
            alerte = False
        else:
            running_cumulative_error += error_seance
            cumul_str = f"{running_cumulative_error:.2f}"
            budget_restant = 10.0 - total_cumulative_error
            
            if error_seance > 0:
                seances_str = str(int(budget_restant / error_seance)) if budget_restant > 0 else "0"
            else:
                seances_str = "Illimité"

            commentaires = []
            alerte = False
            if error_seance > 1.5:
                commentaires.append("Dose > 1.5%")
                alerte = True
            if total_cumulative_error >= 10.0:
                commentaires.append("DÉPASSÉ !")
                alerte = True
            commentaire = " | ".join(commentaires) if commentaires else "OK"
        
        final_table_data.append({
            "Date": item['Date'], "Machine": item['Machine'],
            "Dose / Fraction": f"{item['Dose (Gy)']:.2f}", "uLCT (%)": f"{item['uLCT (%)']:.2f}",
            "Erreur Séance (%)": f"{error_seance:.2f}" if index > 0 else "-",
            "Erreur Cumulée (%)": cumul_str, "Séances Restantes": seances_str,
            "Commentaire": commentaire, "_Alerte": alerte, "_Index": index
        })

    # 3. Affichage visuel
    c1, c2 = st.columns(2)
    c1.info(f"**Patient :** {raw_data[0]['Patient']}")
    c2.info(f"**ID :** {raw_data[0]['ID']}")

    df = pd.DataFrame(final_table_data)
    def style_dataframe(row):
        if row['_Index'] == 0: return ['background-color: #d1e7dd; font-weight: bold'] * len(row)
        elif row['_Alerte']: return ['background-color: #ffebee; color: #d32f2f; font-weight: bold'] * len(row)
        return [''] * len(row)

    styled_df = df.style.apply(style_dataframe, axis=1).hide(['_Alerte', '_Index'], axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    return doublons_ignores, True



st.markdown("<h1 style='text-align: center;'>Suivi des doses Tomo</h1>", unsafe_allow_html=True)

# Création de deux onglets distincts
tab_in, tab_out = st.tabs(["Traiter les nouveaux plans (IN)", " Base de données globale (OUT)"])

# ==========================================
# ONGLET 1 : GESTION DES NOUVEAUX ARRIVAGES
# ==========================================
with tab_in:
    st.markdown(f"<p style='text-align: center;'>Placez vos nouveaux plans dans le dossier <b>{DIR_IN}</b> puis cliquez sur le bouton.</p>", unsafe_allow_html=True)
    
    fichiers_in = lire_fichiers_dossier(DIR_IN)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Traiter les nouveaux plans", use_container_width=True):
            if len(fichiers_in) == 0:
                st.warning(f"Aucun nouveau fichier trouvé dans le dossier {DIR_IN}.")
            else:
                with st.spinner("Analyse des nouveaux plans et récupération des historiques..."):
                    
                    # 1. Identifier les IDs des patients présents dans IN
                    patients_cibles_ids = set()
                    for f in fichiers_in:
                        # Lecture rapide juste pour chopper l'ID
                        plan_temporaire = dcm.dcmread(f, stop_before_pixels=True)
                        patients_cibles_ids.add(str(plan_temporaire.PatientID))
                    
                    # 2. Chercher dans OUT les fichiers appartenant SEULEMENT à ces patients
                    fichiers_out = lire_fichiers_dossier(DIR_OUT)
                    fichiers_historique = []
                    for f in fichiers_out:
                        plan_temporaire = dcm.dcmread(f, stop_before_pixels=True)
                        if str(plan_temporaire.PatientID) in patients_cibles_ids:
                            fichiers_historique.append(f)
                    
                    # 3. Combiner le passé (historique ciblé) et le présent (fichiers IN)
                    fichiers_a_traiter = fichiers_historique + fichiers_in
                    
                    # 4. Générer le tableau
                    doublons_ignores, succes = analyser_et_afficher_tableau(fichiers_a_traiter)
                    
                    # 5. Déplacer les fichiers de IN vers OUT une fois le calcul fini
                    for fichier in fichiers_in:
                        nom_fichier = os.path.basename(fichier)
                        chemin_dest = os.path.join(DIR_OUT, nom_fichier)
                        if os.path.exists(chemin_dest):
                            os.remove(chemin_dest)
                        shutil.move(fichier, DIR_OUT)
                    
                    if succes:
                        if doublons_ignores > 0:
                            st.success(f" Traitement terminé ({doublons_ignores} doublon(s) ignoré(s)). Les fichiers ont été archivés dans OUT.")
                        else:
                            st.success(f"Traitement terminé. Les fichiers ont été archivés dans OUT.")

# ==========================================
# ONGLET 2 : CONSULTATION DE LA BASE DE DONNÉES
# ==========================================
with tab_out:
    st.markdown("### Rechercher l'historique d'un patient")
    
    fichiers_out = lire_fichiers_dossier(DIR_OUT)
    
    if len(fichiers_out) == 0:
        st.info(f"La base de données est vide. Les fichiers traités apparaîtront ici.")
    else:
        # Extraire la liste de tous les patients existants dans OUT
        patients_disponibles = {}
        for f in fichiers_out:
            plan_temporaire = dcm.dcmread(f, stop_before_pixels=True)
            pat_id = str(plan_temporaire.PatientID)
            
            if pat_id not in patients_disponibles:
                nom_brut = str(plan_temporaire.PatientName).split("^")
                nom_propre = f"{nom_brut[1] if len(nom_brut) > 1 else ''} {nom_brut[0]}".strip()
                patients_disponibles[pat_id] = f"{nom_propre} (ID: {pat_id})"
        
        # Créer le menu déroulant avec la liste triée
        liste_choix = sorted(list(patients_disponibles.values()))
        patient_selectionne = st.selectbox("Sélectionnez un patient :", ["-- Choisir un patient --"] + liste_choix)
        
        if patient_selectionne != "-- Choisir un patient --":
            # Extraire l'ID du choix (ce qui est entre les parenthèses)
            id_cible = patient_selectionne.split("ID: ")[1].replace(")", "")
            
            with st.spinner("Chargement de l'historique..."):
                # Ne garder que les fichiers du patient sélectionné
                fichiers_patient = []
                for f in fichiers_out:
                    if str(dcm.dcmread(f, stop_before_pixels=True).PatientID) == id_cible:
                        fichiers_patient.append(f)
                
                # Générer le tableau pour ce patient précis
                analyser_et_afficher_tableau(fichiers_patient)

                #####