#======= Import libraries ===============================================================================
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

#======= Page Configuration ================================================================================
st.set_page_config(page_title="Tomo Transfer", layout="wide")
#===========================================================================================================

#======= Automatic Folder & Database Setup =================================================================
DIR_IN = r"\\nasdata1\TOMO\Transfert_tomo" 
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
        Full_Name TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS SESSIONS (
        Session_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Patient_ID TEXT,
        DICOM_SOP_UID TEXT UNIQUE,
        Date_Time TEXT,
        Display_Date TEXT,
        Machine TEXT,
        Treatment_Site TEXT,
        Raw_Treatment_Site TEXT, -- NOUVELLE COLONNE POUR LE TEXTE INTOUCHABLE
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
    conn.commit()
    conn.close()

init_db()
#===========================================================================================================

#======= Sinogram Extraction ===============================================================================
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

#======= Patient & Machine Info ============================================================================
#======= Patient & Machine Info ============================================================================
#======= Patient & Machine Info ============================================================================
def general_info(plan): 
    plan_info = {} 
    plan_info["patient_id"] = str(plan.PatientID)
    plan_info["patient_name"] = str(plan.PatientName)
    
    # --- 1. ASPIRATEUR DU NOM BRUT ---
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

        # Nettoyage des doublons
        valeurs_uniques = []
        for v in valeurs_trouvees:
            if v not in valeurs_uniques:
                valeurs_uniques.append(v)

        texte_brut = " | ".join(valeurs_uniques) if valeurs_uniques else "Aucun texte trouvé"
        plan_info["raw_site"] = texte_brut
        
    except Exception as e:
        plan_info["raw_site"] = "Erreur lecture"
        texte_brut = ""

    # --- 2. AUTO-CATÉGORISATION (Pour tes futures stats) ---
    texte_minuscule = texte_brut.lower()
    
    if any(mot in texte_minuscule for mot in ["sein", "paroi", "mam"]):
        plan_info["treatment_site"] = "Sein & Paroi"
    elif any(mot in texte_minuscule for mot in ["orl", "larynx", "pharynx", "cavite", "cervico"]):
        plan_info["treatment_site"] = "ORL"
    elif any(mot in texte_minuscule for mot in ["pelvis", "prostate", "rectum", "col", "vagin", "anal", "vessie", "gyneco"]):
        plan_info["treatment_site"] = "Pelvis & Gynéco"
    elif any(mot in texte_minuscule for mot in ["cerveau", "encephale", "crane", "cereb", "stereotaxie"]):
        plan_info["treatment_site"] = "Cérébral"
    elif any(mot in texte_minuscule for mot in ["poumon", "thorax", "mediastin"]):
        plan_info["treatment_site"] = "Thorax & Poumon"
    elif any(mot in texte_minuscule for mot in ["abdomen", "foie", "pancreas", "estomac"]):
        plan_info["treatment_site"] = "Abdomen"
    else:
        plan_info["treatment_site"] = "Inconnu" # Si aucun mot-clé n'est trouvé
    # --------------------------------------------------------------------------------

    serial_mapping = {"4010012": "Tomo2", "210462": "Tomo4", "4010710": "Radi7"} 
    try:
        raw_serial = str(plan.DeviceSerialNumber) 
        plan_info["machine_nb"] = serial_mapping.get(raw_serial, raw_serial)
    except AttributeError:
        plan_info["machine_nb"] = "Unknown"
        
    return plan_info
#===========================================================================================================
#===========================================================================================================
#===========================================================================================================

#======= Physics Parameters Extraction =====================================================================
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

#======= Error Calculation =================================================================================
def get_error_shift(sinogram, delivery):
    PT = delivery["PT"] 
    LOT_sino = PT*sinogram  
    maxLOT = np.max(LOT_sino) 
    total_lot = np.sum(LOT_sino) 
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  
    thresh = 18  
    undisc_LCT = 0
        
    error_per_projection = np.zeros(LOT_sino.shape[0]) 
    
    cond1 = LOT_sino < (maxLOT-1)  
    cond2 = LOT_sino > (PT-thresh) 
    row, col = np.where(cond1 & cond2) 
                                       
    for i in range(len(row)):
        if row[i] < (LOT_sino.shape[0] - 1):             
            if (LOT_sino[row[i]+1, col[i]] > (PT-20)): 
                undisc_LCT += 1 
        else: 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row>(-0.5)] 
    filtered_col = col[col>(-0.5)]
    extra_time = 0  
    
    for i in range(len(filtered_row)-1):
        diff = (PT - LOT_sino[filtered_row[i], filtered_col[i]]) 
        extra_time += diff  
        error_per_projection[filtered_row[i]] += diff 
    
    error_per_projection_pct = (error_per_projection / total_lot) * 100 
    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100), error_per_projection_pct
#===========================================================================================================

#======= File Reading ======================================================================================
def read_files_in_directory(directory):
    files_found = []
    for root, dirs, files in os.walk(directory):  
        for file in files:
            if file.startswith(("RP", "RTPLAN")) and file.endswith(".dcm"):  
                files_found.append(os.path.join(root, file))
    return files_found
#===========================================================================================================

#======= Dashboard Engine ==================================================================================
def display_dashboard(raw_data):
    if not raw_data:
        st.warning("Aucune donnée à afficher pour ce patient.")
        return

    # === Affichage de la Carte Patient ===
    patient_name = raw_data[0]['Patient']
    patient_id = raw_data[0]['ID']
    nb_frac_ref = raw_data[0]['Nb_Frac']
    site_traitement = raw_data[0].get('Site', 'Inconnu') 
    site_brut = raw_data[0].get('Raw_Site', 'Inconnu')
    
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
        
        # Les choix standards propres que tu veux avoir dans ta base au final
        liste_sites_propres = ["Sein & Paroi", "ORL", "Pelvis & Gynéco", "Cérébral", "Thorax & Poumon", "Abdomen", "Autre", "Inconnu"]
        
        # On rajoute la valeur brute dans la liste pour qu'elle s'affiche par défaut si elle n'a pas encore été corrigée
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
                
                # Mise à jour SQL
                conn_update = sqlite3.connect(DB_NAME)
                cursor_update = conn_update.cursor()
                
                cursor_update.execute("""
                    UPDATE SESSIONS 
                    SET Treatment_Site = ? 
                    WHERE Patient_ID = ?
                """, (nouveau_site, patient_id))
                
                conn_update.commit()
                conn_update.close()
                
                st.success("Nom mis à jour !")
                time.sleep(0.5) 
                st.rerun() 
                
    st.markdown("<br>", unsafe_allow_html=True)

    # === Saisie des fractions (Validation par le clinicien) ===
    st.markdown("#####  Validation des séances réalisées")
    st.markdown("<small style='color: #6c757d;'>Le DICOM ne reflétant pas toujours la réalité clinique de Mosaiq/Aria, merci de valider le nombre de séances réellement effectuées.</small>", unsafe_allow_html=True)
    
    fractions_done_list = []
    cols = st.columns(len(raw_data))
    
    for index in range(len(raw_data)):
        machine = raw_data[index]['Machine']
        if index < len(raw_data) - 1:
            # Pour les machines passées : on tente une soustraction automatique par défaut
            diff_auto = int(max(1, raw_data[index]['Nb_Frac'] - raw_data[index+1]['Nb_Frac']))
            with cols[index]:
                f_done = st.number_input(f"Faites sur {machine}", min_value=0, max_value=int(nb_frac_ref), value=diff_auto, key=f"frac_{index}")
            fractions_done_list.append(f_done)
        else:
            # Pour la machine actuelle : on met 1 par défaut (modifiable)
            with cols[index]:
                f_done = st.number_input(f"Aujourd'hui sur {machine}", min_value=0, max_value=int(nb_frac_ref), value=1, key=f"frac_{index}")
            fractions_done_list.append(f_done)
    st.markdown("<br>", unsafe_allow_html=True)

    # === Construction du tableau et calcul des doses cumulées ===
    total_cumulated_dose = 0.0 
    final_table_data = [] 
    
    try:
        dose_nominal_ref = raw_data[0]['Dose (Gy)']
        total_budget_Gy = (dose_nominal_ref * nb_frac_ref) + 0.5 
    except:
        total_budget_Gy = 0.0

    for index, item in enumerate(raw_data):
        error_session_pct = item['Session Error (%)']
        dose_nominal = item['Dose (Gy)']
        fractions_done = fractions_done_list[index]
        
        actual_dose_session = dose_nominal * (1 + (error_session_pct / 100.0)) 
        
        if index == 0:
            date_display = item['Date'] + " (Initiale)" 
            total_cumulated_dose += (dose_nominal * fractions_done) 
            
            cumul_str = f"{total_cumulated_dose:.2f}"
            comment = f"Prescription : {total_budget_Gy:.2f} Gy"
            alert = False
            
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
            
            comments = []
            alert = False
            if error_session_pct > 1.5: comments.append("Erreur > 1.5%"); alert = True
            if total_cumulated_dose >= total_budget_Gy: comments.append("BUDGET DÉPASSÉ"); alert = True
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
    
    col_graph1, col_graph2 = st.columns([1, 1])

    with col_graph1:
        st.subheader("Consommation du Budget Dose") 
        if len(df) > 0:
            last_session_df = df.iloc[-1]
            budget_max =  last_session_df['_Budget_Total']
            current_dose = last_session_df['_Cumul_Val']

            percentage = min(current_dose / budget_max, 1.0) if budget_max > 0 else 0.0
            
            st.progress(percentage)
            st.markdown(f"<h3 style='text-align: center; color: #333; margin-bottom: 0px;'>{current_dose:.2f} Gy / {budget_max:.2f} Gy</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 13px; color: #6c757d; margin-top: 0px;'><i>(Dose prescrite + 0.5 Gy de tolérance)</i></p>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### Simulateur de fin de traitement")

            if len(df) > 1: 
                initial_machine = df.iloc[0]['Machine']
                unique_machines = [m for m in df['Machine'].unique().tolist() if m != initial_machine]
                
                if len(unique_machines) > 0:
                    machine_chosen = st.radio("Projeter la suite du traitement avec :", options=unique_machines, horizontal=True)
                    
                    last_session_machine = df[df['Machine'] == machine_chosen].iloc[-1]
                    
                    if  last_session_machine['Dose Délivrée (Gy/s.)'] == "-":
                        simulated_dose = float(last_session_machine['Dose Prévue (Gy/s.)'])
                    else:
                        simulated_dose = float(last_session_machine['Dose Délivrée (Gy/s.)'])
                    
                    budget_remaining = budget_max - current_dose
                    max_sessions = int(budget_remaining / simulated_dose) if simulated_dose > 0 and budget_remaining > 0 else 0
                    
                    # === CALCUL DU TEXTE HISTORIQUE ===
                    breakdown_dict = {}
                    for idx in range(len(raw_data)):
                        mach = raw_data[idx]['Machine']
                        f = fractions_done_list[idx]
                        if f > 0:
                            breakdown_dict[mach] = breakdown_dict.get(mach, 0) + f
                            
                    total_done_simul = sum(fractions_done_list)
                    
                    if total_done_simul > 0:
                        breakdown_str = " <i>(" + ", ".join([f"{count} avec la {m}" for m, count in breakdown_dict.items()]) + ")</i>"
                    else:
                        breakdown_str = ""
                    
                    theoretical_remaining_sessions = nb_frac_ref - total_done_simul
                    # ==================================
                    
                    if max_sessions > 0:
                        if max_sessions == theoretical_remaining_sessions:
                            bg_color = "#d4edda" 
                            text_color = "#155724"
                            alert_loss = ""
                        else:
                            bg_color = "#f8d7da" 
                            text_color = "#721c24"
                            if max_sessions < theoretical_remaining_sessions:
                                loss = theoretical_remaining_sessions - max_sessions
                                alert_loss = f" <i>(soit une perte de <b>{loss} séance(s)</b>)</i>"
                            else:
                                gain = max_sessions - theoretical_remaining_sessions
                                alert_loss = f" <i>(soit un décalage de <b>+{gain} séance(s)</b>)</i>"
                            
                        html_text = f"""
                        <div style="background-color: {bg_color}; padding: 16px; border-radius: 8px; color: {text_color}; margin-top: 10px;">
                            <b>Si le patient continue sur {machine_chosen} :</b><br><br>
                            Il a déjà réalisé <b>{total_done_simul} séance(s)</b>{breakdown_str}.<br><br>
                            Au rythme de cette machine (<b>{simulated_dose:.2f} Gy/séance</b>), il peut encore faire <b>{max_sessions} séances</b> au lieu des <b>{theoretical_remaining_sessions}</b> restantes prévues{alert_loss}.
                        </div>
                        """
                        st.markdown(html_text, unsafe_allow_html=True)
                    else:
                        st.error(f"ALERTE CRITIQUE : Le budget total sera dépassé à la prochaine séance sur la {machine_chosen} !")
                else:
                    st.info("Le traitement est actuellement sur la machine initiale. Aucun transfert n'a encore été enregistré pour simuler une projection.")
            else:
                dose_last = float(last_session_df['Dose Prévue (Gy/s.)'])
                st.info(f"Plan initial : Rythme nominal de {dose_last:.2f} Gy/séance. Le patient doit faire {nb_frac_ref} séances au total.")

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
                    caption_text = f"Dose excess on rotation {rotation_target} (Gantry 0° - 360°)."
                else:
                    slice_errors = total_errors
                    caption_text = "Dose excess over 1 rotation (Gantry 0° - 360°)."
                
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
                
                angles_deg = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                ax.set_xticks(angles_deg) 
                labels_10deg = [f"{i}°" for i in range(0, 360, 10)]
                ax.set_xticklabels(labels_10deg, fontsize=7, color='#2c3e50') 

                ax.set_facecolor('white') 
                tick_values = [max_err_global * 0.25, max_err_global * 0.5, max_err_global * 0.75, max_err_global]
                ax.set_yticks(tick_values) 
                
                labels_ticks = [f"{val:.3f}%" for val in tick_values]
                ax.set_yticklabels(labels_ticks, fontsize=7, color='#d32f2f', fontweight='bold') 
                
                ax.set_rlabel_position(25)
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=False)
                st.caption(caption_text)
            else:
                st.info("No error detected in the selected session.")
#===========================================================================================================

#======= BARRE LATÉRALE (Menu Ingestion) ===================================================================
st.sidebar.image("logo.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Service de Physique Médicale**")
st.sidebar.markdown("---")
st.sidebar.markdown("###  Ingestion des plans")
st.sidebar.markdown(f"<small>Traitez les fichiers DICOM déposés dans le dossier **Transfert_tomo** pour les intégrer à la base.</small>", unsafe_allow_html=True)

files_in = read_files_in_directory(DIR_IN)

if st.sidebar.button("Traiter les nouveaux plans", use_container_width=True, type="primary"):
    if len(files_in) == 0:
        st.sidebar.warning(f"Aucun nouveau fichier trouvé.")
    else:
        with st.sidebar.status("Analyse en cours...", expanded=True) as status:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            
            duplicates_ignored = 0
            new_processed = 0

            for f in files_in:
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
                              info["treatment_site"], info["raw_site"], # <--- MODIFICATION ICI
                              delivery["DS"], delivery.get("Nb_Frac", 1), data[1], data[0], profile_json,
                              delivery["CS"], delivery["GP"]))
                        
                        new_processed += 1
                        
                    except sqlite3.IntegrityError:
                        duplicates_ignored += 1
                        
                except Exception as e:
                    st.sidebar.error(f"Erreur sur {os.path.basename(f)} : {e}")
                
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
            st.sidebar.success(f"**{new_processed}** nouveau(x) plan(s) ajouté(s).")
        else:
            st.sidebar.info(f"Aucun ajout. **{duplicates_ignored}** doublon(s) archivé(s).")
#===========================================================================================================

#======= MAIN SCREEN (Clinical Analysis) ===================================================================
st.markdown("<h1 style='text-align: center;'>Suivi de transfert tomo</h1>", unsafe_allow_html=True) 

st.markdown("### Rechercher l'historique d'un patient")

conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()
cursor.execute("SELECT Patient_ID, Full_Name FROM PATIENTS")
patients_db = cursor.fetchall()

if len(patients_db) == 0:
    st.info(f"La base de données est vide. Déposez des fichiers dans le dossier **{DIR_IN}** et cliquez sur le bouton à gauche.")
else:
    list_choices = sorted([f"{p[1]} (ID: {p[0]})" for p in patients_db])

    new_icon_url = "https://img.icons8.com/?size=25&id=7eX13e1GI7bn&format=png&color=000000"
    st.markdown(f' <img src="{new_icon_url}" style="height: 20px; vertical-align: middle;"> Rechercher par Nom, Prénom ou ID :', unsafe_allow_html=True)
    search = st.text_input("", placeholder="Ex: Dupont, Jean, ou 12345...")

    if search:
        list_choices = [p for p in list_choices if search.lower() in p.lower()] 
                
    if len(list_choices) == 0: 
        st.warning("Aucun patient ne correspond à cette recherche.")
    else:
        patient_selected = st.selectbox("Sélectionnez un patient :", ["-- Choisir un patient --"] + list_choices) 
        
        if patient_selected != "-- Choisir un patient --":
            id_target = patient_selected.split("ID: ")[1].replace(")", "")
            
            with st.spinner("Récupération rapide depuis la base SQL..."): 
                cursor.execute("""
                    SELECT s.Display_Date, s.Machine, s.Dose_Gy, s.Nb_Frac, s.uLCT, s.Error_pct, s.Profile_JSON, s.Date_Time, p.Full_Name, s.CS_mm_s, s.GP_s, s.Treatment_Site, s.Raw_Treatment_Site 
                    FROM SESSIONS s
                    JOIN PATIENTS p ON s.Patient_ID = p.Patient_ID
                    WHERE s.Patient_ID = ? 
                    ORDER BY s.Date_Time ASC
                """, (id_target,))
                
                session_lines = cursor.fetchall()
                
                raw_data_sql = []
                for row in session_lines:
                    raw_data_sql.append({
                        'Date': row[0], 'Machine': row[1], 'Dose (Gy)': row[2], 'Nb_Frac': row[3],
                        'uLCT (%)': row[4], 'Session Error (%)': row[5], 'Profile_Error': json.loads(row[6]),
                        'sort_key': row[7], 'Patient': row[8], 'ID': id_target,
                        'CS': row[9], 'GP': row[10],
                        'Site': row[11],
                        'Raw_Site': row[12]
                    })
                    
                display_dashboard(raw_data_sql)
