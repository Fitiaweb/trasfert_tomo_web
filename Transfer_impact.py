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
DIR_IN = r"\\nasdata1\TOMO\Transfert_tomo" # Le 'r' bloque le piège des antislashs
DIR_ARCHIVE = "ARCHIVES"
DB_NAME = "tomo_database.db" #il creer la base de donnée tout seul 

os.makedirs(DIR_IN, exist_ok=True)  #creer le dossier IN s'il n'existe pas, et ne fait rien s'il existe déjà
os.makedirs(DIR_ARCHIVE, exist_ok=True) #creer le dossier ARCHIVES s'il n'existe pas, et ne fait rien s'il existe déjà

    #les tableaux sont en if not exists, donc ils ne seront créés qu'une seule fois, même si la fonction est appelée plusieurs fois.
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
        Dose_Gy REAL,
        Nb_Frac INTEGER,
        uLCT REAL,
        Error_pct REAL,
        Profile_JSON TEXT,
        CS_mm_s REAL,    -- NOUVELLE COLONNE
        GP_s REAL,       -- NOUVELLE COLONNE
        FOREIGN KEY (Patient_ID) REFERENCES PATIENTS(Patient_ID)
    )
    ''')
    conn.commit()
    conn.close()

init_db()
#===========================================================================================================

#======= Sinogram Extraction ===============================================================================
def get_sinogram(plan):
    NCP = plan.BeamSequence[0].NumberOfControlPoints #Nombre de points de contrôle (NCP)
    sinogram = np.zeros((NCP,64)) 
    cp_sequence = plan.BeamSequence[0].ControlPointSequence 

    for cp in range(NCP): 
        try : 
            tmp = cp_sequence[cp][0x300d,0x10a7].value #On récupère la valeur brute du sinogramme pour ce point de contrôle
            tmp = tmp.decode('utf-8').strip('\x00').split('\\') #convertir 
            sinogram[cp-1,:] = np.array(tmp,dtype=np.float64) #On remplit la ligne correspondante du sinogramme
        except KeyError:
            continue 
    return sinogram 
#===========================================================================================================

#======= Patient & Machine Info ============================================================================
def general_info(plan): 
    plan_info = {} 
    plan_info["patient_id"] = str(plan.PatientID)
    plan_info["patient_name"] = str(plan.PatientName)
    serial_mapping = {"4010012": "Tomo2", "210462": "Tomo4", "4010710": "Tomo7"} 
    
    try:
        raw_serial = str(plan.DeviceSerialNumber) 
        plan_info["machine_nb"] = serial_mapping.get(raw_serial, raw_serial)
    except AttributeError:
        plan_info["machine_nb"] = "Unknown"
    return plan_info
#===========================================================================================================

#======= Physics Parameters Extraction =====================================================================
def delivery_info(plan): 
    delivery = {}
    delivery["GP"] = float(plan.BeamSequence[0][0x300d,0x1040].value) #Gantry Period (GP) en secondes
    delivery["PT"] = (delivery["GP"]/51.0)*1000.0   #temps d'une projection en ms
    delivery["CS"] = float(plan.BeamSequence[0][0x300d,0x1080].value) #vitesse de la table en mm/s
    delivery["pitch"] = float(plan.BeamSequence[0][0x300d,0x1060].value) #pitch = (vitesse_table * GP)
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    delivery["Nrot"] = (NCP-1)/51   #nombre de rotations complètes
    delivery["TT"] = delivery["Nrot"]*delivery["GP"] #temps totale de traitement en secondes
    
    try:
        delivery["DS"] = float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose) #Dose prescrite par séance en Gy
        delivery["Nb_Frac"] = int(plan.FractionGroupSequence[0].NumberOfFractionsPlanned) #Nombre de séances prévues par le médecin
    except Exception:
        delivery["DS"] = 0.0
        delivery["Nb_Frac"] = 1
    return delivery
#===========================================================================================================

#======= Error Calculation =================================================================================
def get_error_shift(sinogram, delivery):
    PT = delivery["PT"] 
    LOT_sino = PT*sinogram  #On convertit le sinogramme en LOT (Leaf Opening Time) en multipliant par le temps d'une projection (PT)
    maxLOT = np.max(LOT_sino) #temps max ouvert 
    total_lot = np.sum(LOT_sino) #somme des LOT
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  # On prend uniquement les LOT non nul 
    thresh = 18  
    undisc_LCT = 0
        
    error_per_projection = np.zeros(LOT_sino.shape[0]) 
    
    cond1 = LOT_sino < (maxLOT-1)  #lames totalement ouvert 
    cond2 = LOT_sino > (PT-thresh) #lames ouvert mais superieur au seuil
    row, col = np.where(cond1 & cond2) #row = numero projection et col = numero de la lame
                                       
    for i in range(len(row)):
        if row[i] < (LOT_sino.shape[0] - 1):             
            if (LOT_sino[row[i]+1, col[i]] > (PT-20)): # Vérifie détection de latence moteur MLC
                undisc_LCT += 1 
        else: 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row>(-0.5)] 
    filtered_col = col[col>(-0.5)]
    extra_time = 0  
    
    for i in range(len(filtered_row)-1):
        diff = (PT - LOT_sino[filtered_row[i], filtered_col[i]]) #calcul de l'excès de temps d'ouverture pour cette projection et cette lame
        extra_time += diff  #temps total d'excès d'ouverture accumulé sur tout le traitement
        error_per_projection[filtered_row[i]] += diff 
    
    error_per_projection_pct = (error_per_projection / total_lot) * 100 
    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100), error_per_projection_pct
#===========================================================================================================

#======= File Reading ======================================================================================
def read_files_in_directory(directory):
    files_found = []
    for root, dirs, files in os.walk(directory):  
        for file in files:
            # On accepte désormais les fichiers commençant par "RP" OU "RTPLAN"
            if file.startswith(("RP", "RTPLAN")) and file.endswith(".dcm"):  #au cas ou si le dossier à un autre nom il faudra modifier 
                files_found.append(os.path.join(root, file))
    return files_found
#===========================================================================================================

#======= Dashboard Engine ==================================================================================
def display_dashboard(raw_data):
    if not raw_data:
        st.warning("Aucune donnée à afficher pour ce patient.")
        return

    total_cumulated_dose = 0.0 #initialize cumulative dose variable
    final_table_data = [] 
    
    try:
        dose_nominal_ref = raw_data[0]['Dose (Gy)']
        nb_frac_ref = raw_data[0]['Nb_Frac']
        total_budget_Gy = (dose_nominal_ref * nb_frac_ref) + 0.5 #SEUIL MODIFIABLE SELON LA TOLERANCE ACCEPTABLE (ex: 0.5 Gy)
    except:
        total_budget_Gy = 0.0
        nb_frac_ref = 1

    for index, item in enumerate(raw_data):
        error_session_pct = item['Session Error (%)']
        dose_nominal = item['Dose (Gy)']
        
        actual_dose_session = dose_nominal * (1 + (error_session_pct / 100.0)) #calcul de la dose réelle délivrée en tenant compte de l'erreur de séance
        
        if index == 0:
            date_display = item['Date'] + " (Initiale)" #on marque la première séance pour la différencier visuellement
            total_cumulated_dose += dose_nominal #pour la première séance, on considère que la dose délivrée correspond à la dose prescrite
            cumul_str = f"{total_cumulated_dose:.2f}"
            comment = f"Budget prescript : {total_budget_Gy:.2f} Gy"
            alert = False
            running_cumul_val = total_cumulated_dose
            
            final_table_data.append({
                "Date": date_display, "Machine": item['Machine'],
                "Planned Dose (Gy)": f"{dose_nominal:.2f}", 
                "uLCT (%)": f"{item['uLCT (%)']:.2f}",
                "Session Error (%)": "-",
                "Delivered Dose (Gy)": "-", 
                "Cumulated Dose (Gy)": cumul_str, 
                "Comment": comment, 
                "_Alert": alert, "_Index": index, "_Cumul_Val": running_cumul_val, "_Budget_Total": total_budget_Gy
            })
        else:
            total_cumulated_dose += actual_dose_session
                
            running_cumul_val = total_cumulated_dose
            cumul_str = f"{total_cumulated_dose:.2f}"
            
            comments = []
            alert = False
            if error_session_pct > 1.5: comments.append("Erreur > 1.5%"); alert = True
            if total_cumulated_dose >= total_budget_Gy: comments.append("BUDGET DEPASSE"); alert = True
            comment = " | ".join(comments) if comments else "OK"
        
            final_table_data.append({
                "Date": item['Date'], "Machine": item['Machine'],
                "Planned Dose (Gy)": f"{dose_nominal:.2f}", 
                "uLCT (%)": f"{item['uLCT (%)']:.2f}",
                "Session Error (%)": f"{error_session_pct:.2f}",
                "Delivered Dose (Gy)": f"{actual_dose_session:.2f}",
                "Cumulated Dose (Gy)": cumul_str, 
                "Comment": comment, 
                "_Alert": alert, "_Index": index, "_Cumul_Val": running_cumul_val, "_Budget_Total": total_budget_Gy
            })

    # === Affichage de la Carte Patient (Version 3 Colonnes sans emoji) ===
    patient_name = raw_data[0]['Patient']
    patient_id = raw_data[0]['ID']
    
    html_header = f"""
    <div style="background-color: #f8f9fa; padding: 15px 25px; border-radius: 8px; border-left: 6px solid #1f77b4; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Dossier Patient</span>
            <h2 style="margin: 5px 0 0 0; color: #2c3e50; font-size: 26px;">{patient_name}</h2>
        </div>
        <div style="text-align: center;">
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Séances prévues par le médecin</span>
            <h3 style="margin: 5px 0 0 0; color: #2c3e50; font-size: 24px; font-weight: 700;">{nb_frac_ref}</h3>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 13px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Identifiant (ID)</span>
            <h3 style="margin: 5px 0 0 0; color: #1f77b4; font-size: 22px;"># {patient_id}</h3>
        </div>
    </div>
    """
    st.markdown(html_header, unsafe_allow_html=True)

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
                elif row['Machine'] == 'Tomo7': cell_style = 'background-color: #c8e6c9; color: #000000; font-weight: bold;' 
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
            "_Budget_Total": None 
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
                # On isole la machine de la séance initiale (première ligne du tableau)
                initial_machine = df.iloc[0]['Machine']
                # On filtre la liste des machines uniques pour exclure totalement la machine initiale
                unique_machines = [m for m in df['Machine'].unique().tolist() if m != initial_machine]
                
                if len(unique_machines) > 0:
                    machine_chosen = st.radio("Projeter la suite du traitement avec :", options=unique_machines, horizontal=True)
                    
                    last_session_machine = df[df['Machine'] == machine_chosen].iloc[-1]
                    
                    if  last_session_machine['Delivered Dose (Gy)'] == "-":
                        simulated_dose = float(last_session_machine['Planned Dose (Gy)'])
                    else:
                        simulated_dose = float(last_session_machine['Delivered Dose (Gy)'])
                    
                    budget_remaining = budget_max - current_dose
                    max_sessions = int(budget_remaining / simulated_dose) if simulated_dose > 0 and budget_remaining > 0 else 0
                    
                    sessions_done = len(df)
                    theoretical_remaining_sessions = nb_frac_ref - sessions_done
                    
                    if max_sessions > 0:
                        # Logique de coloration dynamique (Vert si ça ne bouge pas, Rouge si ça change)
                        if max_sessions == theoretical_remaining_sessions:
                            bg_color = "#d4edda" # Vert (Stable)
                            text_color = "#155724"
                            alert_loss = ""
                        else:
                            bg_color = "#f8d7da" # Rouge (Décalage de dose)
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
                            Il a déjà réalisé <b>{sessions_done} séance(s)</b>.<br><br>
                            Au rythme de cette machine (<b>{simulated_dose:.2f} Gy</b>), il peut encore faire <b>{max_sessions} séances</b> au lieu des <b>{theoretical_remaining_sessions}</b> initialement prévues{alert_loss}.
                        </div>
                        """
                        st.markdown(html_text, unsafe_allow_html=True)
                    else:
                        st.error(f"ALERTE CRITIQUE : Le budget total sera dépassé à la prochaine séance sur la {machine_chosen} !")
                else:
                    st.info("Le traitement est actuellement sur la machine initiale. Aucun transfert n'a encore été enregistré pour simuler une projection.")
            else:
                dose_last = float(last_session_df['Planned Dose (Gy)'])
                st.info(f"Plan initial : Rythme nominal de {dose_last:.2f} Gy/séance. Le patient doit faire {nb_frac_ref} séances au total.")

    with col_graph2:
        st.subheader("Angular Localization") 
        
        # 1. Identify ONLY true transfer sessions (ignoring the initial baseline)
        transfer_sessions = []
        if len(raw_data) > 0:
            previous_machine = raw_data[0]['Machine']
            
            for item in raw_data[1:]: # We start looking from the second session
                current_machine = item['Machine']
                if current_machine != previous_machine:
                    transfer_sessions.append(item)
                    previous_machine = current_machine
        
        # 2. Check if any transfers actually happened
        if len(transfer_sessions) == 0:
            st.info("No machine transfer detected for this patient. The treatment remained on the initial machine.")
        else:
            # 3. Create labels for the dropdown menu (only real transfers)
            menu_options = []
            for s in transfer_sessions:
                menu_options.append(f"Transfer: {s['Date']} to {s['Machine']}")
                    
            # 4. Display the dropdown menu to the user
            selected_date = st.selectbox("Select the transfer event to analyze:", options=menu_options)
            
            # 5. Retrieve data for the selected session
            chosen_index = menu_options.index(selected_date)
            session_to_analyze = transfer_sessions[chosen_index]

            if session_to_analyze['Session Error (%)'] > 0.0:
                total_errors = np.array(session_to_analyze['Profile_Error'])
                
                # Retrieval of physical parameters (with safety if missing)
                CS = session_to_analyze.get('CS', 0) # couch speed in mm/s
                GP = session_to_analyze.get('GP', 0) # gantry period in seconds
                
                distance_tour_cm = (CS * GP) / 10.0 if CS and GP else 0  # Table advancement per rotation in cm
                
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
                
                # Security if length is not exactly a multiple of 51
                N_total = len(slice_errors)
                angles = np.linspace(0, 2 * np.pi, N_total, endpoint=False) 
                
                angles_closed = np.concatenate((angles, [angles[0]])) 
                errors_closed = np.concatenate((slice_errors, [slice_errors[0]])) 
                
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5, 5.5)) 
                ax.set_theta_zero_location("N") 
                ax.set_theta_direction(-1) 
      
                max_err_global = np.max(total_errors) # Y-scale is fixed on the max error of the WHOLE treatment
                if max_err_global == 0:
                    max_err_global = 0.1
                ax.set_ylim(max_err_global * 1.3, 0) 

                ax.fill_between(angles_closed, 0, errors_closed, color='#ff4757', alpha=0.35) 
                ax.plot(angles_closed, errors_closed, color='#c0392b', linewidth=2.0, zorder=3)
                
                # Apply degree labels
                angles_deg = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                ax.set_xticks(angles_deg) 
                labels_10deg = [f"{i}°" for i in range(0, 360, 10)]
                ax.set_xticklabels(labels_10deg, fontsize=7, color='#2c3e50') 

                ax.set_facecolor('white') 
                tick_values = [max_err_global * 0.25, max_err_global * 0.5, max_err_global * 0.75, max_err_global]
                ax.set_yticks(tick_values) 
                
                # Create text for labels (e.g., "1.5%")
                labels_ticks = [f"{val:.3f}%" for val in tick_values]
                ax.set_yticklabels(labels_ticks, fontsize=7, color='#d32f2f', fontweight='bold') 
                
                # Shift value axis to 25 degrees so it doesn't overlap
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
                                Patient_ID, DICOM_SOP_UID, Date_Time, Display_Date, Machine, 
                                Dose_Gy, Nb_Frac, uLCT, Error_pct, Profile_JSON, CS_mm_s, GP_s
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (info["patient_id"], uid, date_time_sort, display_date, info["machine_nb"], 
                              delivery["DS"], delivery.get("Nb_Frac", 1), data[1], data[0], profile_json,
                              delivery["CS"], delivery["GP"])) # <-- AJOUT DES VARIABLES ICI
                        
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
st.markdown("<h1 style='text-align: center;'>Suivi des doses Tomo</h1>", unsafe_allow_html=True) 

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
                    SELECT s.Display_Date, s.Machine, s.Dose_Gy, s.Nb_Frac, s.uLCT, s.Error_pct, s.Profile_JSON, s.Date_Time, p.Full_Name, s.CS_mm_s, s.GP_s 
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
                        'CS': row[9], 'GP': row[10] 
                    })
                    
                display_dashboard(raw_data_sql)