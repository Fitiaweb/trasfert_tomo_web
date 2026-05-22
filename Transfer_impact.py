#======= Import bibliothèque ===============================================================================

import streamlit as st
import pydicom as dcm
import numpy as np
import pandas as pd
import io

# Configuration de la page web
st.set_page_config(page_title="QA Radiothérapie - Tomo", layout="wide")

st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("---") # Ajoute une petite ligne de séparation en dessous
st.sidebar.markdown("**Service de Physique Médicale**")

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



# Streamlit gère l'upload de fichiers automatiquement avec un beau bouton "Drag & Drop" !
uploaded_files = st.file_uploader("Glissez vos fichiers DICOM (.dcm) ici", accept_multiple_files=True)

if uploaded_files:
    raw_data = []
    plans_vus = set()
    doublons_ignores = 0
    
    # 1. Extraction des données
    with st.spinner("Analyse des projections en cours..."):
        for file in uploaded_files:
            # Pydicom sait lire directement depuis le fichier web sans l'enregistrer sur le PC
            plan = dcm.dcmread(io.BytesIO(file.read()))
            
            sinogram = get_sinogram(plan) 
            info = general_info(plan) 
            delivery = delivery_info(plan) 
            data = get_error_shift(sinogram, delivery) 
            
            parts = info["patient_name"].split("^")  
            name_id = f"{parts[1] if len(parts) > 1 else ''} {parts[0]}".strip()
            
            try: plan_date = str(plan[0x0008, 0x0012].value)
            except KeyError: plan_date = "00000000" 
            try: plan_time = str(plan[0x0008, 0x0013].value)
            except KeyError: plan_time = "000000"

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

    # Tri chronologique
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
            if error_seance > 1.5:
                commentaires.append("Dose/séance > 1.5%")
                alerte = True
            if total_cumulative_error >= 10.0:
                commentaires.append("Seuil 10% DÉPASSÉ !")
                alerte = True
                
            commentaire = " | ".join(commentaires) if commentaires else "OK"
        
        final_table_data.append({
            "Date": item['Date'], "Machine": item['Machine'],
            "Dose / Fraction": f"{item['Dose (Gy)']:.2f}", "uLCT (%)": f"{item['uLCT (%)']:.2f}",
            "Erreur Séance (%)": f"{error_seance:.2f}" if index > 0 else "-",
            "Erreur Cumulée (%)": cumul_str,
            "Séances Restantes": seances_str,
            "Commentaire": commentaire,
            "_Alerte": alerte,
            "_Index": index
        })

    # 3. Création du tableau interactif avec Pandas
    df = pd.DataFrame(final_table_data)

    # Fonction pour colorier le tableau (L'équivalent des tags dans Tkinter)
    def style_dataframe(row):
        if row['_Index'] == 0:
            return ['background-color: #d1e7dd; font-weight: bold'] * len(row)
        elif row['_Alerte']:
            return ['background-color: #ffebee; color: #d32f2f; font-weight: bold'] * len(row)
        else:
            return [''] * len(row)

    # On applique le style et on enlève les colonnes de logique interne
    styled_df = df.style.apply(style_dataframe, axis=1).hide(['_Alerte', '_Index'], axis=1)

    # Affichage du tableau magnifique sur la page web
    st.dataframe(styled_df, use_container_width=True, height=400)