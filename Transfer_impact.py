#======= Import bibliothèque ===============================================================================

import streamlit as st
import pydicom as dcm
import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt

#===========================================================================================================



#======= Configuration de la page web ======================================================================

st.set_page_config(page_title="Transfert Tomo", layout="wide") #titre page 

st.sidebar.image("logo.png", use_container_width=True) #logo ( test )
st.sidebar.markdown("---") #barre de séparation 
st.sidebar.markdown("**Service de Physique Médicale**") #texte en bas de la barre latérale

#===========================================================================================================





#======= Création automatique des dossiers IN et OUT  ======================================================

DIR_IN = "IN"
DIR_OUT = "OUT"
os.makedirs(DIR_IN, exist_ok=True)
os.makedirs(DIR_OUT, exist_ok=True)

#===========================================================================================================






#=======  Récupération sinogrammes    ======================================================================

def get_sinogram(plan):
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    sinogram = np.zeros((NCP,64)) 
    cp_sequence = plan.BeamSequence[0].ControlPointSequence 
    for cp in range(NCP): 
        try : 
            tmp = cp_sequence[cp][0x300d,0x10a7].value 
            # CORRECTIF 1 : Nettoyage de l'octet vide DICOM
            tmp = tmp.decode('utf-8').strip('\x00').split('\\')
            sinogram[cp-1,:] = np.array(tmp,dtype=np.float64)  
        except KeyError:
            continue 
    return sinogram 

#===========================================================================================================









#=======  Récupération Nom + Prénom + Machine     ==========================================================

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








#=======  Récupération Gantry + Projection + couch speed+ pitch + nb rot + temps de traitement + dose  =====

def delivery_info(plan): 
    delivery = {}
    delivery["GP"] = float(plan.BeamSequence[0][0x300d,0x1040].value) #gantry periode
    delivery["PT"] = (delivery["GP"]/51.0)*1000.0   #Projection Time (ms)
    delivery["CS"] = float(plan.BeamSequence[0][0x300d,0x1080].value) #Couch Speed (mm/s)
    delivery["pitch"] = float(plan.BeamSequence[0][0x300d,0x1060].value) #Pitch
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    delivery["Nrot"] = (NCP-1)/51   #PNumber of gantry rotations  
    delivery["TT"] = delivery["Nrot"]*delivery["GP"] #Treatment Time (s)
    
    try:
        delivery["DS"] = float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose) #Dose delivered (Gy)
    except Exception:
        delivery["DS"] = 0.0
    return delivery

#===========================================================================================================








#=======  Récupération erreur de dose en fct de projection    ==============================================

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

    # transforme tableau en pourcentage d'erreur
    erreur_par_projection_pct = (erreur_par_projection / total_lot) * 100


    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100), erreur_par_projection_pct

#===========================================================================================================





#=======  Lire les fichiers    =============================================================================

def lire_fichiers_dossier(dossier):
    fichiers_trouves = []
    for root, dirs, files in os.walk(dossier):  
        for file in files:
            if file.startswith("RP") and file.endswith(".dcm"):  
                fichiers_trouves.append(os.path.join(root, file))
    return fichiers_trouves

#===========================================================================================================






#=======  Grand tableau   ==================================================================================

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
            'Erreur Séance (%)': data[0],
            'Profil_Erreur': data[2]
        })

    if not raw_data:
        return doublons_ignores, False

    raw_data.sort(key=lambda x: x['sort_key'])
    
    total_cumulative_error = 0.0
    for index, item in enumerate(raw_data):
        if index > 0: total_cumulative_error += item['Erreur Séance (%)']

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
            running_cumul_val = 0.0
        else:
            running_cumulative_error += error_seance
            running_cumul_val = running_cumulative_error
            cumul_str = f"{running_cumulative_error:.2f}"
            budget_restant = 10.0 - total_cumulative_error
            seances_str = str(int(budget_restant / error_seance)) if error_seance > 0 and budget_restant > 0 else "0" if error_seance > 0 else "Illimité"
            
            commentaires = []
            alerte = False
            if error_seance > 1.5: commentaires.append("Dose > 1.5%"); alerte = True
            if total_cumulative_error >= 10.0: commentaires.append("DÉPASSÉ !"); alerte = True
            commentaire = " | ".join(commentaires) if commentaires else "OK"
        
        final_table_data.append({
            "Date": item['Date'], "Machine": item['Machine'],
            "Dose / Fraction": f"{item['Dose (Gy)']:.2f}", "uLCT (%)": f"{item['uLCT (%)']:.2f}",
            "Erreur Séance (%)": f"{error_seance:.2f}" if index > 0 else "-",
            "Erreur Cumulée (%)": cumul_str, "Séances Restantes": seances_str,
            "Commentaire": commentaire, "_Alerte": alerte, "_Index": index, "_Cumul_Val": running_cumul_val
        })

#===========================================================================================================






#=======   Afichage des deuc graphes   =====================================================================

    c1, c2 = st.columns(2)
    c1.info(f"**Patient :** {raw_data[0]['Patient']}")
    c2.info(f"**ID :** {raw_data[0]['ID']}")

    df = pd.DataFrame(final_table_data)
    
    def style_dataframe(row):
        if row['_Index'] == 0: return ['background-color: #d1e7dd; font-weight: bold'] * len(row)
        elif row['_Alerte']: return ['background-color: #ffebee; color: #d32f2f; font-weight: bold'] * len(row)
        return [''] * len(row)

    # Masquage des colonnes techniques (_Alerte, _Index, _Cumul_Val)
    styled_df = df.style.apply(style_dataframe, axis=1).hide(['_Alerte', '_Index', '_Cumul_Val'], axis=1)
    st.dataframe(styled_df, use_container_width=True, height=200)

    st.markdown("---")
    col_graph1, col_graph2 = st.columns([1, 1])

    with col_graph1:
        st.subheader(" Évolution de l'Erreur Cumulée")
        if len(df) > 1:
            df_trend = df.copy()
            df_trend['Séance'] = ["Séance " + str(i+1) for i in range(len(df_trend))]
            df_trend['Limite Max (10%)'] = 10.0
            chart_data = df_trend.set_index('Séance')[['_Cumul_Val', 'Limite Max (10%)']]
            chart_data.rename(columns={'Erreur Cumulée (%)': 'Erreur Cumulée (%)'}, inplace=True)
            st.line_chart(chart_data, color=["#29b5e8", "#FF0000"])
        else:
            st.info("Une seule séance (référence).")

    with col_graph2:
        st.subheader(" Localisation angulaire (Tomo-vue)")
        derniere_seance = raw_data[-1]
        if derniere_seance['Erreur Séance (%)'] > 0.0:
            erreurs = derniere_seance['Profil_Erreur']
            angles = np.linspace(0, 2*np.pi, len(erreurs))
            
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.fill(angles, erreurs, color='red', alpha=0.6)
            ax.plot(angles, erreurs, color='red', linewidth=1)
            st.pyplot(fig)
            st.caption("Distribution de l'excès de dose par angle de projection.")
        else:
            st.success("Aucune erreur détectée sur la dernière séance.")
            
    return doublons_ignores, True

#===========================================================================================================







#======= Création de deux onglets  =========================================================================

st.markdown("<h1 style='text-align: center;'>Suivi des doses Tomo</h1>", unsafe_allow_html=True)

tab_in, tab_out = st.tabs(["Traiter les nouveaux plans (IN)", " Base de données globale (OUT)"])

#===========================================================================================================







#======= gestion des nouveaux plans ========================================================================

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
                    
                    # 4. Générer le tableau et les graphiques
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
                            st.success(f" Traitement terminé ({doublons_ignores} doublon(s) ignoré(s)). Les fichiers ont été archivés.")
                        else:
                            st.success(f"Traitement terminé. Les fichiers ont été archivés dans OUT.")


#===========================================================================================================






#======= Gestion des anciens plans  ========================================================================

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
            id_cible = patient_selectionne.split("ID: ")[1].replace(")", "")
            
            with st.spinner("Chargement de l'historique..."):
                fichiers_patient = []
                for f in fichiers_out:
                    if str(dcm.dcmread(f, stop_before_pixels=True).PatientID) == id_cible:
                        fichiers_patient.append(f)
                
                # Générer le tableau et les graphiques pour ce patient précis
                analyser_et_afficher_tableau(fichiers_patient)

#===========================================================================================================

