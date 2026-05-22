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
            N_total = len(erreurs) # Nombre total de points dans le traitement
            
            # Angles continus pour tracer la courbe entière
            angles = np.linspace(0, 2 * np.pi, N_total, endpoint=False)
            
            # On ferme la boucle pour le tracé
            angles_fermes = np.concatenate((angles, [angles[0]]))
            erreurs_fermees = np.concatenate((erreurs, [erreurs[0]]))
            
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5, 5.5))
            
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            
            max_err = np.max(erreurs)
            ax.set_ylim(max_err * 1.4, 0)
            
            # Tracé de la courbe (utilise tous les points)
            ax.fill_between(angles_fermes, 0, erreurs_fermees, color='#FF4B4B', alpha=0.7)
            ax.plot(angles_fermes, erreurs_fermees, color='red', linewidth=1.5)
            
            # --- LE CORRECTIF EST ICI ---
            # On génère exactement 51 positions pour les graduations
            angles_51 = np.linspace(0, 2 * np.pi, 51, endpoint=False)
            ax.set_xticks(angles_51)
            
            # On affiche les numéros de 1 à 51 sur ces positions (taille de police 6 pour ne pas surcharger)
            ax.set_xticklabels([str(i+1) for i in range(51)], fontsize=6)
            
            # Masquer le fond gris s'il y en a un et retirer les labels radiaux
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
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Traiter les nouveaux plans", use_container_width=True):
            if len(fichiers_in) == 0:
                st.warning(f"Aucun nouveau fichier trouvé dans le dossier *IN*.")
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
                            st.success(f" Traitement terminé (**{doublons_ignores}** doublon(s) ignoré(s)). Les fichiers ont été archivés.")
                        else:
                            st.success(f"Traitement terminé. Les fichiers ont été archivés dans OUT.")


#===========================================================================================================






#======= Gestion des anciens plans  ========================================================================

#======= Gestion des anciens plans  ========================================================================

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
        
        # Créer le menu déroulant avec la liste globale triée
        liste_choix = sorted(list(patients_disponibles.values()))
        
        # --- Barre de recherche ---
        
        # 1. On définit l'URL de la nouvelle icône (taille 25px pour un label)
        new_icon_url = "https://img.icons8.com/?size=25&id=7eX13e1GI7bn&format=png&color=000000"

        # 2. On affiche l'icône et le texte d'invite avec Markdown et HTML non sécurisé
        # L'image est chargée en ligne, pas besoin de fichier local, comme demandé.
        st.markdown(f' <img src="{new_icon_url}" style="height: 20px; vertical-align: middle;"> Rechercher par Nom, Prénom ou ID :', unsafe_allow_html=True)
        
        # 3. Le champ d'entrée de texte n'a plus de label direct, il utilise le placeholder
        recherche = st.text_input("", placeholder="Ex: Dupont, Jean, ou 12345...")
        
        # Filtrer la liste si du texte est entré (insensible à la casse avec .lower())
        if recherche:
            liste_choix = [p for p in liste_choix if recherche.lower() in p.lower()]
            
        # Si la recherche ne donne rien
        if len(liste_choix) == 0:
            st.warning("Aucun patient ne correspond à cette recherche.")
        else:
            # Menu déroulant avec la liste (filtrée ou complète) conservant le choix par défaut
            patient_selectionne = st.selectbox("Sélectionnez un patient :", ["-- Choisir un patient --"] + liste_choix)
            
            if patient_selectionne != "-- Choisir un patient --":
                # Extraction de l'ID pour retrouver les fichiers DICOM
                id_cible = patient_selectionne.split("ID: ")[1].replace(")", "")
                
                with st.spinner("Chargement de l'historique..."):
                    fichiers_patient = []
                    for f in fichiers_out:
                        if str(dcm.dcmread(f, stop_before_pixels=True).PatientID) == id_cible:
                            fichiers_patient.append(f)
                    
                    # Générer le tableau et les graphiques pour ce patient précis
                    analyser_et_afficher_tableau(fichiers_patient)

#===========================================================================================================



#=============== Suppression dossier out ===================================================================

#suppression des dossier du out pour eviter saturation du site et des données 

    st.markdown("---")
    st.markdown("###  Gestion de la base de données")

    # 1. Initialisation de l'état de confirmation si il n'existe pas encore
    if 'demande_suppression' not in st.session_state:
        st.session_state.demande_suppression = False

    # 2. Premier bouton : Déclencheur
    if not st.session_state.demande_suppression:
        if st.button("Supprimer le contenu du dossier *OUT*", use_container_width=True):
            st.session_state.demande_suppression = True
            st.rerun() # On relance pour afficher l'étape suivante

    # 3. Étape de double vérification (ne s'affiche que si le bouton 1 a été cliqué)
    if st.session_state.demande_suppression:
        st.warning("**Double vérification demandée**")
        
        # Champ de saisie
        phrase = st.text_input("Veuillez entrer la phrase **oui supprimer** pour déverrouiller l'action :")
        
        col_annuler, col_valider = st.columns(2)
        
        with col_annuler:
            if st.button("Annuler", use_container_width=True):
                st.session_state.demande_suppression = False
                st.rerun()

        with col_valider:
            # Le bouton final de suppression n'est cliquable que si la phrase est exacte
            if phrase == "oui supprimer":
                if st.button("CONFIRMER LA SUPPRESSION DÉFINITIVE", type="primary", use_container_width=True):
                    try:
                        # Suppression des fichiers
                        for filename in os.listdir(DIR_OUT):
                            file_path = os.path.join(DIR_OUT, filename)
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        
                        # Succès et réinitialisation
                        st.success(" Dossier OUT vidé avec succès !")
                        st.session_state.demande_suppression = False
                        # On attend un petit peu pour que l'utilisateur voie le message de succès avant de rafraîchir
                        import time
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            else:
                # Bouton grisé/désactivé tant que la phrase n'est pas bonne
                st.button("CONFIRMER LA SUPPRESSION ", disabled=True, use_container_width=True)





#===========================================================================================================



#===========  JAUGE DE STOCKAGE / PERFORMANCES     =========================================================

    st.markdown("---")
    st.markdown(" #### État de la base active (Performances)")
    
    # 1. On compte le nombre réel de fichiers DICOM archivés dans OUT
    nb_fichiers_out = len(lire_fichiers_dossier(DIR_OUT))
    LIMITE_MAX = 500
    
    # 2. Calcul du pourcentage pour la barre (plafonné à 1.0 maximum pour Streamlit)
    pourcentage = min(nb_fichiers_out / LIMITE_MAX, 1.0)
    
    # 3. Affichage de la barre de progression
    st.progress(pourcentage)
    
    # 4. Message dynamique avec alertes selon le volume de données
    if nb_fichiers_out >= LIMITE_MAX:
        st.error(f" **Seuil critique atteint ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** Les performances de recherche et d'affichage sont dégradées. Veuillez vider le dossier OUT avant les prochains traitements.")
    elif nb_fichiers_out >= (LIMITE_MAX * 0.8): # À partir de 400 fichiers
        st.warning(f" **Volume élevé ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** Pensez à vider le dossier prochainement pour maintenir une fluidité maximale dans le service.")
    else:
        st.success(f" **Système optimal ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** La lecture des données et la génération des graphiques sont instantanées.")



#===========================================================================================================