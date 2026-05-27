#======= Import bibliothèque ===============================================================================
import streamlit as st # pour créer l'interface web 
import pydicom as dcm # pour lire les fichiers DICOM et extraire les informations nécessaires
import numpy as np # pour tout les calculs 
import pandas as pd # pour la gestion des données et l'affichage du tableau
import os # gestion des dossier 
import shutil # pour déplacer les fichiers de IN vers OUT après traitement
import matplotlib.pyplot as plt # pour tracer le graphique polaire
import time # pour les pauses d'affichage
#===========================================================================================================

#======= Configuration de la page web ======================================================================
st.set_page_config(page_title="Transfert Tomo", layout="wide") #titre page 
st.sidebar.image("logo.png", use_container_width=True) #logo ( test )
st.sidebar.markdown("---") #barre de séparation 
st.sidebar.markdown("**Service de Physique Médicale**") #texte en bas de la barre latérale
#===========================================================================================================

#======= Création automatique des dossiers IN, OUT et ARCHIVES =============================================
DIR_IN = "IN" # Ou ton chemin réseau si tu l'as changé
DIR_OUT = "OUT"
DIR_ARCHIVE = "ARCHIVES"
os.makedirs(DIR_IN, exist_ok=True) 
os.makedirs(DIR_OUT, exist_ok=True)
os.makedirs(DIR_ARCHIVE, exist_ok=True)
#========================================================================================

#=======  Récupération sinogrammes    ======================================================================
def get_sinogram(plan):
    NCP = plan.BeamSequence[0].NumberOfControlPoints  #nombre de contrôle points
    sinogram = np.zeros((NCP,64)) #creer un sinogramme vide de la taille NCP x 64 
    cp_sequence = plan.BeamSequence[0].ControlPointSequence #récupérer la séquence des contrôle points du plan

    for cp in range(NCP): 
        try : 
            tmp = cp_sequence[cp][0x300d,0x10a7].value #récup^ère les valeurs des sinogramme dans le dicom 
            tmp = tmp.decode('utf-8').strip('\x00').split('\\') # Nettoyage de l'octet vide DICOM
            sinogram[cp-1,:] = np.array(tmp,dtype=np.float64)  #range les valeurs dans le sinogramme vide
        except KeyError:
            continue 
    return sinogram 
#===========================================================================================================

#=======  Récupération Nom + Prénom + Machine  =============================================================
def general_info(plan): 
    plan_info = {} #stcoker les infos plans 
    plan_info["patient_id"] = str(plan.PatientID)
    plan_info["patient_name"] = str(plan.PatientName)
    serial_mapping = {"4010012": "Tomo2", "210462": "Tomo4", "4010710": "Tomo7"} # mapping des tomo
    
    try:
        raw_serial = str(plan.DeviceSerialNumber) #récupérer le numéro de série
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
        delivery["Nb_Frac"] = int(plan.FractionGroupSequence[0].NumberOfFractionsPlanned) # Nombre de séances prévues
    except Exception:
        delivery["DS"] = 0.0
        delivery["Nb_Frac"] = 1
    return delivery
#===========================================================================================================

#=======  Récupération erreur de dose en fct de projection    ==============================================
def get_error_shift(sinogram,delivery):
    PT = delivery["PT"] # projection time 
    LOT_sino = PT*sinogram  #transforme mon sinogramm en sinogramme de LOT (Leaf Open Time) en ms
    maxLOT = np.max(LOT_sino) #extrait la valeur la plus grande du sinogramem de LOT
    total_lot = np.sum(LOT_sino) #somme tous les LOT
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  #enleve tout les 0 de ma matrice de LOT
    thresh = 18  #seuil de 18ms 
    undisc_LCT = 0
        
    erreur_par_projection = np.zeros(LOT_sino.shape[0]) #creer une matrice pour stocker les erreurs de porjection
    
    cond1 = LOT_sino < (maxLOT-1)  #enlever toute les ouvertures de 100% 
    cond2 = LOT_sino > (PT-thresh) #si LOT>282ms alors il y aura une erreur de dose
    row,col = np.where(cond1 & cond2) #row = num de projection avec erreur de lame
                                      #col = num de lame pas ferme   
    for i in range(len(row)):
        if row[i] < (LOT_sino.shape[0] - 1):             
            if (LOT_sino[row[i]+1,col[i]] > (PT-20)): #20ms car on tolère un 2ms de + (donc 18 + 2 = 20) 
                undisc_LCT += 1 
        else: 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row>(-0.5)] 
    filtered_col = col[col>(-0.5)]
    extra_time = 0  
    
    for i in range(len(filtered_row)-1):
        diff = (PT - LOT_sino[filtered_row[i],filtered_col[i]])
        extra_time += diff  #temsps d'extra ouvertures de lames pour toute la seance 
        erreur_par_projection[filtered_row[i]] += diff #temps d'extra ouvertures de lames pour chaque projection (51 projections)
    
    erreur_par_projection_pct = (erreur_par_projection / total_lot) * 100 # transforme tableau de ms -> en pourcentage d'erreur
    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100), erreur_par_projection_pct
#===========================================================================================================

#=======  Lire les fichiers    =============================================================================
def lire_fichiers_dossier(dossier):
    fichiers_trouves = []
    for root, dirs, files in os.walk(dossier):  
        for file in files:
            if file.startswith("RP") and file.endswith(".dcm"):  #si le dossier est bien un RP et que c'est un fichier DICOM
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
            'Nb_Frac': delivery.get("Nb_Frac", 1), # On récupère le nombre de fractions
            'uLCT (%)': data[1],
            'Erreur Séance (%)': data[0],
            'Profil_Erreur': data[2]
        })

    if not raw_data:
        return doublons_ignores, False

    raw_data.sort(key=lambda x: x['sort_key']) 
    
    dose_cumulee_totale = 0.0
    final_table_data = []
    
    # Le budget total est défini par le plan de référence
    try:
        dose_nominale_ref = raw_data[0]['Dose (Gy)']
        nb_frac_ref = raw_data[0]['Nb_Frac']
        budget_total_Gy = (dose_nominale_ref * nb_frac_ref) + 0.5 #CHANGER SEUIL ICI 
    except:
        budget_total_Gy = 0.0

    for index, item in enumerate(raw_data):
        error_seance_pct = item['Erreur Séance (%)']
        dose_nominale = item['Dose (Gy)']
        
        dose_reelle_seance = dose_nominale * (1 + (error_seance_pct / 100.0))
        
        if index == 0:
            item['Date'] += " (Initiale)"
            dose_cumulee_totale += dose_nominale 
            budget_restant = budget_total_Gy - dose_cumulee_totale
            seances_max_possibles = int(budget_restant / dose_nominale) if dose_nominale > 0 else 0
            
            cumul_str = f"{dose_cumulee_totale:.2f}"
            commentaire = f"Budget prescript : {budget_total_Gy:.2f} Gy"
            alerte = False
            running_cumul_val = dose_cumulee_totale
            
            final_table_data.append({
                "Date": item['Date'], "Machine": item['Machine'],
                "Dose Prévue (Gy)": f"{dose_nominale:.2f}", 
                "uLCT (%)": f"{item['uLCT (%)']:.2f}",
                "Erreur Séance (%)": "-",
                "Dose Délivrée (Gy)": "-", 
                "Dose Cumulée (Gy)": cumul_str, 
                "Séances Max Possibles": str(seances_max_possibles),
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
                "Séances Max Possibles": str(seances_max_possibles),
                "Commentaire": commentaire, 
                "_Alerte": alerte, "_Index": index, "_Cumul_Val": running_cumul_val, "_Budget_Total": budget_total_Gy
            })

#===========================================================================================================
#=======  Afichage des deux graphes   ======================================================================
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
                if row['Machine'] == 'Tomo2':
                    cell_style = 'background-color: #bbdefb; color: #000000; font-weight: bold;' 
                elif row['Machine'] == 'Tomo4':
                    cell_style = 'background-color: #ffcdd2; color: #000000; font-weight: bold;' 
                elif row['Machine'] == 'Tomo7':
                    cell_style = 'background-color: #c8e6c9; color: #000000; font-weight: bold;' 
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
    
    # === Graphe 1 pour évolution de la Dose envoyé  ===
    col_graph1, col_graph2 = st.columns([1, 1])

    with col_graph1:
        st.subheader(" Consommation du Budget Dose") 
        if len(df) > 0:
            derniere_seance_df = df.iloc[-1]
            budget_max = derniere_seance_df['_Budget_Total']
            dose_actuelle = derniere_seance_df['_Cumul_Val']

            pourcentage = min(dose_actuelle / budget_max, 1.0) if budget_max > 0 else 0.0
            
            st.progress(pourcentage)
            st.markdown(f"<h3 style='text-align: center; color: #333;'>{dose_actuelle:.2f} Gy / {budget_max:.2f} Gy</h3>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("####  Simulateur de fin de traitement")

            # --- 2. LOGIQUE DES BOUTONS DE PROJECTION ---
            if len(df) > 1: # S'il y a plusieurs séances, on affiche les boutons
                
                # CORRECTION : On prend TOUTES les machines du tableau, y compris la première
                machines_uniques = df['Machine'].unique().tolist()
                
                # Création du bouton cliquable (Radio horizontal)
                machine_choisie = st.radio(
                    "Projeter la suite du traitement avec :",
                    options=machines_uniques,
                    horizontal=True
                )
                
                # Trouver la TOUTE DERNIÈRE séance faite sur CETTE machine sélectionnée
                derniere_seance_machine = df[df['Machine'] == machine_choisie].iloc[-1]
                
                # CORRECTION : Si la machine choisie est celle de la séance initiale (qui a un tiret "-"), on prend la dose prévue
                if derniere_seance_machine['Dose Délivrée (Gy)'] == "-":
                    dose_simulee = float(derniere_seance_machine['Dose Prévue (Gy)'])
                else:
                    dose_simulee = float(derniere_seance_machine['Dose Délivrée (Gy)'])
                
                # Nouveaux calculs dynamiques basés sur la sélection
                budget_restant = budget_max - dose_actuelle
                seances_max = int(budget_restant / dose_simulee) if dose_simulee > 0 and budget_restant > 0 else 0
                
                seances_faites = len(df)
                nb_frac_ref = raw_data[0]['Nb_Frac']
                seances_restantes_theoriques = nb_frac_ref - seances_faites
                
                # Affichage du résultat de la simulation avec couleur dynamique
                if seances_max > 0:
                    perte = seances_restantes_theoriques - seances_max
                    # On passe en balises HTML <b> et <i> pour que ça marche dans la div
                    alerte_perte = f" <i>(soit une perte de <b>{perte} séance(s)</b>)</i>" if perte > 0 else ""
                    
                    # Définition de la couleur de fond
                    couleur_fond = "#f0f8ff" # Par défaut
                    if machine_choisie == 'Tomo2':
                        couleur_fond = "#bbdefb" # Bleu
                    elif machine_choisie == 'Tomo4':
                        couleur_fond = "#ffcdd2" # Rouge
                    elif machine_choisie == 'Tomo7':
                        couleur_fond = "#c8e6c9" # Vert
                        
                    # Création du bloc de texte en HTML
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
            

            
            # --- 3. CAS OÙ IL N'Y A QUE LE PLAN INITIAL ---
            else:
                dose_derniere = float(derniere_seance_df['Dose Prévue (Gy)'])
                nb_frac_ref = raw_data[0]['Nb_Frac']
                st.info(f" **Plan initial :** Rythme nominal de {dose_derniere:.2f} Gy/séance. Le patient doit faire **{nb_frac_ref} séances** au total.")
        else:
            st.info("Aucune donnée disponible.")

    # === FIN DU NOUVEAU BLOC ===

    with col_graph2:
        st.subheader(" Localisation angulaire (Tomo-vue)")
        derniere_seance = raw_data[-1] 

        if derniere_seance['Erreur Séance (%)'] > 0.0:
            erreurs = derniere_seance['Profil_Erreur']
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
            
    return doublons_ignores, True
#===========================================================================================================

#======= Création de deux onglets  =========================================================================
st.markdown("<h1 style='text-align: center;'>Suivi des doses Tomo</h1>", unsafe_allow_html=True) 
tab_in, tab_out = st.tabs(["Traiter les nouveaux plans (*IN*)", " Base de données globale (*OUT*)"]) 
#===========================================================================================================

#======= gestion des nouveaux plans ========================================================================
with tab_in:
    st.markdown(f"<p style='text-align: center;'>Placez vos nouveaux plans dans le dossier <b>{DIR_IN}</b> puis cliquez sur le bouton.</p>", unsafe_allow_html=True)
    
    fichiers_in = lire_fichiers_dossier(DIR_IN)
    
    col1, col2, col3 = st.columns([1, 3, 1]) 

    with col2:
        lancement = st.button("Traiter les nouveaux plans", use_container_width=True)
        
    if lancement:
        if len(fichiers_in) == 0:
            st.warning(f"Aucun nouveau fichier trouvé dans le dossier *IN*.")
        else:
            with st.spinner("Analyse des nouveaux plans et récupération des historiques..."): 
                
                patients_cibles_ids = set()  
                for f in fichiers_in:
                    plan_temporaire = dcm.dcmread(f, stop_before_pixels=True) 
                    patients_cibles_ids.add(str(plan_temporaire.PatientID))
                
                fichiers_out = lire_fichiers_dossier(DIR_OUT)
                fichiers_historique = []
                for f in fichiers_out:
                    plan_temporaire = dcm.dcmread(f, stop_before_pixels=True)
                    if str(plan_temporaire.PatientID) in patients_cibles_ids:
                        fichiers_historique.append(f)
               
                fichiers_a_traiter = fichiers_historique + fichiers_in
                
                doublons_ignores, succes = analyser_et_afficher_tableau(fichiers_a_traiter)
                            
                for fichier in fichiers_in: 
                    nom_fichier = os.path.basename(fichier)
                    chemin_dest = os.path.join(DIR_OUT, nom_fichier)
                    if os.path.exists(chemin_dest):
                        os.remove(chemin_dest)
                    shutil.move(fichier, DIR_OUT)
                
                if succes:
                    if doublons_ignores > 0:
                        st.success(f" Traitement terminé (**{doublons_ignores}** doublon(s) ignoré(s)). Les fichiers ont été archivés.")
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
        patients_disponibles = {}
        for f in fichiers_out:
            plan_temporaire = dcm.dcmread(f, stop_before_pixels=True)
            pat_id = str(plan_temporaire.PatientID) 
            
            if pat_id not in patients_disponibles:
                nom_brut = str(plan_temporaire.PatientName).split("^")
                nom_propre = f"{nom_brut[1] if len(nom_brut) > 1 else ''} {nom_brut[0]}".strip()
                patients_disponibles[pat_id] = f"{nom_propre} (ID: {pat_id})"
        
        liste_choix = sorted(list(patients_disponibles.values())) 

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
                
                with st.spinner("Chargement de l'historique..."): 
                    fichiers_patient = []
                    for f in fichiers_out:
                        if str(dcm.dcmread(f, stop_before_pixels=True).PatientID) == id_cible:
                            fichiers_patient.append(f)
                    analyser_et_afficher_tableau(fichiers_patient)
#===========================================================================================================

#=============== Archivage dossier out ===================================================================
    st.markdown("---")
    st.markdown("###  Gestion de la base de données")

    # On change la variable de session pour l'archivage
    if 'demande_archivage' not in st.session_state: 
        st.session_state.demande_archivage = False

    if not st.session_state.demande_archivage:
        if st.button("Archiver le contenu du dossier *OUT*", use_container_width=True): 
            st.session_state.demande_archivage = True
            st.rerun() 

    if st.session_state.demande_archivage: 
        st.warning("**Double vérification demandée (Les fichiers n'apparaîtront plus dans l'historique)**")
        phrase = st.text_input("Veuillez entrer la phrase **oui archiver** pour déverrouiller l'action :")
        col_annuler, col_valider = st.columns(2)
        
        with col_annuler:
            if st.button("Annuler", use_container_width=True):
                st.session_state.demande_archivage = False
                st.rerun()

        with col_valider:
            if phrase == "oui archiver":
                if st.button("CONFIRMER L'ARCHIVAGE", type="primary", use_container_width=True):
                    try:
                        for filename in os.listdir(DIR_OUT): 
                            file_path = os.path.join(DIR_OUT, filename)
                            if os.path.isfile(file_path):
                                chemin_dest = os.path.join(DIR_ARCHIVE, filename)
                                if os.path.exists(chemin_dest):
                                    os.remove(chemin_dest)
                                shutil.move(file_path, DIR_ARCHIVE)
                                
                        st.success(" Dossier OUT vidé et fichiers sauvegardés dans ARCHIVES !") 
                        st.session_state.demande_archivage = False
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            else:
                st.button("CONFIRMER L'ARCHIVAGE ", disabled=True, use_container_width=True)
#===========================================================================================================

#===========  JAUGE DE STOCKAGE / PERFORMANCES   =========================================================
    st.markdown("---")
    st.markdown(" #### État du dossier OUT ")  
    nb_fichiers_out = len(lire_fichiers_dossier(DIR_OUT)) 
    LIMITE_MAX = 500
    pourcentage = min(nb_fichiers_out / LIMITE_MAX, 1.0) 
    st.progress(pourcentage) 
    if nb_fichiers_out >= LIMITE_MAX:
        st.error(f" **Seuil critique atteint ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** Les performances de recherche et d'affichage sont dégradées. Veuillez vider le dossier OUT avant les prochains traitements.")
    elif nb_fichiers_out >= (LIMITE_MAX * 0.8): 
        st.warning(f" **Volume élevé ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** Pensez à vider le dossier prochainement pour maintenir une fluidité maximale dans le service.")
    else:
        st.success(f" **Système optimal ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** La lecture des données et la génération des graphiques sont instantanées.")
#===========================================================================================================