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
os.makedirs(DIR_IN, exist_ok=True) #fonctionne comme mkdir 
os.makedirs(DIR_OUT, exist_ok=True)
#===========================================================================================================
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
#=======  Récupération Nom + Prénom + Machine     ==========================================================
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
    except Exception:
        delivery["DS"] = 0.0
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
        
        parts = info["patient_name"].split("^")  #en dicom les noms sont au format "NOM^Prénom"
        name_id = f"{parts[1] if len(parts) > 1 else ''} {parts[0]}".strip() #on inverse pour avoir "Prénom NOM"
        
        try: plan_date = str(plan[0x0008, 0x0012].value) #récupérer la date du plan (format AAAAMMJJ)
        except KeyError: plan_date = "00000000" 
        try: plan_time = str(plan[0x0008, 0x0013].value) #récupérer l'heure du plan (format HHMMSS)
        except KeyError: plan_time = "000000"
        
        cle_unique = f"{info['patient_id']}_{plan_date}_{plan_time}" #creer uen clé unique pour eviter les doublons
        
        if cle_unique in plans_vus:
            doublons_ignores += 1 #compter les doublons ignorés
            continue
        
        plans_vus.add(cle_unique) #ajouter la clé unique à l'ensemble des plans vus pour les futurs vérifications de doublons
        
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

    raw_data.sort(key=lambda x: x['sort_key']) #trier les données par date et heure de planification pour afficher dans l'ordre chronologique
    
    total_cumulative_error = 0.0
    for index, item in enumerate(raw_data):
        if index > 0: total_cumulative_error += item['Erreur Séance (%)'] #calculer l'erreur cumulée totale

    running_cumulative_error = 0.0 #
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
    c1, c2 = st.columns(2) #créer deux colonnes pour afficher les infos patient et ID à côté du tableau
    c1.info(f"**Patient :** {raw_data[0]['Patient']}")
    c2.info(f"**ID :** {raw_data[0]['ID']}")

    df = pd.DataFrame(final_table_data) 
    
    def style_dataframe(row):
        if row['_Index'] == 0: return ['background-color: #f0f8ff ; font-weight: bold'] * len(row) #mettre en évidence plan init
        elif row['_Alerte']: return ['background-color: #ffebee; color: #d32f2f; font-weight: bold'] * len(row) #mettre en évidence les séances avec alerte
        return [''] * len(row)
    styled_df = df.style.apply(style_dataframe, axis=1) 
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        height=200,
        column_config={
            "_Alerte": None,      # None masque complètement la colonne dans l'interface
            "_Index": None,
            "_Cumul_Val": None
        }
    )
    st.markdown("---")
    col_graph1, col_graph2 = st.columns([1, 1])

    with col_graph1:
        st.subheader(" Évolution de l'Erreur Cumulée") #titre du graphique
        if len(df) > 1:
            df_trend = df.copy() #créer une copie du dataframe pour le graphique
            df_trend['Séance'] = ["Séance " + str(i+1) for i in range(len(df_trend))] #créer une colonne pour chaque séances 
            df_trend['Limite Max (10%)'] = 10.0 #ajouter une colonne pour la limite de 10% d'erreur cumulée
            chart_data = df_trend.set_index('Séance')[['_Cumul_Val', 'Limite Max (10%)']] 
            st.line_chart(chart_data, color=["#29b5e8", "#FF0000"]) #
        else:
            st.info("Une seule séance (référence).")

    with col_graph2:
        st.subheader(" Localisation angulaire (Tomo-vue)")
        derniere_seance = raw_data[-1] #récupère les données de la dernière séance pour afficher le profil d'erreur de dose en fonction de l'angle de projection

        if derniere_seance['Erreur Séance (%)'] > 0.0:
            erreurs = derniere_seance['Profil_Erreur']
            N_total = len(erreurs) # Nombre total de points dans le traitement
            angles = np.linspace(0, 2 * np.pi, N_total, endpoint=False) #découper le cercle en N point (2pi = 360°) 
            
            # On ferme la boucle pour le tracé
            angles_fermes = np.concatenate((angles, [angles[0]])) #fermer le cercle en mettant la derniere valeur d'angle égale à la première
            erreurs_fermees = np.concatenate((erreurs, [erreurs[0]])) #fermer le cercle en mettant la derniere valeur d'angle égale à la première
            
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5.5, 5.5)) #créer un graphique polaire pour afficher les erreurs en fonction de l'angle de projection
            
            ax.set_theta_zero_location("N") #mettre 0° en haut du cercle (Nord)
            ax.set_theta_direction(-1) #faire tourner les angles dans le sens des aiguilles d'une montre
            
            max_err = np.max(erreurs) #trouver la valeur maximale d'erreur pour ajuster les limites du graphique 
            ax.set_ylim(max_err * 1.4, 0) #inverser l'axe radial pour que les erreurs plus grandes soient plus proches du centre et ajouter un peu de marge (1.4) pour que le graphique soit plus lisible

            ax.fill_between(angles_fermes, 0, erreurs_fermees, color='#FF4B4B', alpha=0.7) # colorier la zone sous la courbe 
            ax.plot(angles_fermes, erreurs_fermees, color='red', linewidth=1.5) # tracer la courbe d'erreur avec une ligne rouge plus épaisse
            
            angles_51 = np.linspace(0, 2 * np.pi, 51, endpoint=False)
            ax.set_xticks(angles_51) 
            
            ax.set_xticklabels([str(i+1) for i in range(51)], fontsize=6) #afficher les angles 

            ax.set_facecolor('white') #force l'interieur du graphique à être blanc pour une meilleure lisibilité ( mode sombre )
            ax.set_yticks([max_err * 0.25, max_err * 0.5, max_err * 0.75, max_err]) #
            ax.set_yticklabels([]) 
            
            plt.tight_layout()
            
            st.pyplot(fig, use_container_width=False)
            st.caption("Excès de dose ramené sur 1 rotation (51 projections).")
        else:
            st.success("Aucune erreur détectée sur la dernière séance.")
            
    return doublons_ignores, True
#===========================================================================================================
#======= Création de deux onglets  =========================================================================
st.markdown("<h1 style='text-align: center;'>Suivi des doses Tomo</h1>", unsafe_allow_html=True) #titre centré
tab_in, tab_out = st.tabs(["Traiter les nouveaux plans (*IN*)", " Base de données globale (*OUT*)"]) #deux onglets 
#===========================================================================================================
#======= gestion des nouveaux plans ========================================================================
with tab_in:
    st.markdown(f"<p style='text-align: center;'>Placez vos nouveaux plans dans le dossier <b>{DIR_IN}</b> puis cliquez sur le bouton.</p>", unsafe_allow_html=True)
    
    fichiers_in = lire_fichiers_dossier(DIR_IN)
    
    col1, col2, col3 = st.columns([1, 3, 1]) #technique pour centrer le bouton en utilisant des colonnes vides de chaque côté

    with col2:
        lancement = st.button("Traiter les nouveaux plans", use_container_width=True)
    if lancement:
        if len(fichiers_in) == 0:
            st.warning(f"Aucun nouveau fichier trouvé dans le dossier *IN*.")
        else:
            with st.spinner("Analyse des nouveaux plans et récupération des historiques..."): #chargement pendant le traitement
                
                patients_cibles_ids = set() # = ensemble donc refuse les doublons 
                for f in fichiers_in:
                    plan_temporaire = dcm.dcmread(f, stop_before_pixels=True) #je veux juste lire le PAtiendID et non tout le fichier
                    patients_cibles_ids.add(str(plan_temporaire.PatientID))
                
                fichiers_out = lire_fichiers_dossier(DIR_OUT)
                fichiers_historique = []
                for f in fichiers_out:
                    plan_temporaire = dcm.dcmread(f, stop_before_pixels=True)
                    if str(plan_temporaire.PatientID) in patients_cibles_ids:
                        fichiers_historique.append(f)
               
                fichiers_a_traiter = fichiers_historique + fichiers_in
                
                doublons_ignores, succes = analyser_et_afficher_tableau(fichiers_a_traiter)
                            
                for fichier in fichiers_in: # Déplacer les fichiers de IN vers OUT
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
            pat_id = str(plan_temporaire.PatientID) #passer le num patient en chaine de caractère
            
            if pat_id not in patients_disponibles:
                nom_brut = str(plan_temporaire.PatientName).split("^")
                nom_propre = f"{nom_brut[1] if len(nom_brut) > 1 else ''} {nom_brut[0]}".strip()
                patients_disponibles[pat_id] = f"{nom_propre} (ID: {pat_id})"
        
        liste_choix = sorted(list(patients_disponibles.values())) #menu deroulant de tout les patients dans le OUT

        new_icon_url = "https://img.icons8.com/?size=25&id=7eX13e1GI7bn&format=png&color=000000"
        st.markdown(f' <img src="{new_icon_url}" style="height: 20px; vertical-align: middle;"> Rechercher par Nom, Prénom ou ID :', unsafe_allow_html=True)
        recherche = st.text_input("", placeholder="Ex: Dupont, Jean, ou 12345...")

        if recherche:
            liste_choix = [p for p in liste_choix if recherche.lower() in p.lower()] #filtre les choix en fonction de la rechercher llower -> DUPONT -> dupont
                   
        if len(liste_choix) == 0: # Si la recherche ne donne rien
            st.warning("Aucun patient ne correspond à cette recherche.")
        else:
            
            patient_selectionne = st.selectbox("Sélectionnez un patient :", ["-- Choisir un patient --"] + liste_choix) # Menu déroulant avec la liste (filtrée ou complète) conservant le choix par défaut
            
            if patient_selectionne != "-- Choisir un patient --":
                id_cible = patient_selectionne.split("ID: ")[1].replace(")", "")
                
                with st.spinner("Chargement de l'historique..."): # chargement pendant le traitement
                    fichiers_patient = []
                    for f in fichiers_out:
                        if str(dcm.dcmread(f, stop_before_pixels=True).PatientID) == id_cible:
                            fichiers_patient.append(f)
                    analyser_et_afficher_tableau(fichiers_patient)
#===========================================================================================================
#=============== Suppression dossier out ===================================================================
    st.markdown("---")
    st.markdown("###  Gestion de la base de données")

    if 'demande_suppression' not in st.session_state: #Initialisation de l'état de confirmation si il n'existe pas encore
        st.session_state.demande_suppression = False

    if not st.session_state.demande_suppression:
        if st.button("Supprimer le contenu du dossier *OUT*", use_container_width=True): # Premier bouton : Déclencheur
            st.session_state.demande_suppression = True
            st.rerun() # On relance pour afficher l'étape suivante

    if st.session_state.demande_suppression: # Étape de double vérification (ne s'affiche que si le bouton 1 a été cliqué)
        st.warning("**Double vérification demandée**")
        phrase = st.text_input("Veuillez entrer la phrase **oui supprimer** pour déverrouiller l'action :")
        col_annuler, col_valider = st.columns(2)
        
        with col_annuler:
            if st.button("Annuler", use_container_width=True):
                st.session_state.demande_suppression = False
                st.rerun()

        with col_valider:
            if phrase == "oui supprimer":
                if st.button("CONFIRMER LA SUPPRESSION DÉFINITIVE", type="primary", use_container_width=True):
                    try:
                        for filename in os.listdir(DIR_OUT): # Suppression des fichiers
                            file_path = os.path.join(DIR_OUT, filename)
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                        st.success(" Dossier OUT vidé avec succès !") # Succès et réinitialisation
                        st.session_state.demande_suppression = False
                        import time
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Erreur : {e}")
            else:
                st.button("CONFIRMER LA SUPPRESSION ", disabled=True, use_container_width=True)
#===========================================================================================================
#===========  JAUGE DE STOCKAGE / PERFORMANCES     =========================================================
    st.markdown("---")
    st.markdown(" #### État du dossier OUT ")  
    nb_fichiers_out = len(lire_fichiers_dossier(DIR_OUT)) #On compte le nombre réel de fichiers DICOM archivés dans OUT
    LIMITE_MAX = 500
    pourcentage = min(nb_fichiers_out / LIMITE_MAX, 1.0) # calcul du pourcentage pour la barre
    st.progress(pourcentage) # Affichage de la barre de progression
    if nb_fichiers_out >= LIMITE_MAX:
        st.error(f" **Seuil critique atteint ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** Les performances de recherche et d'affichage sont dégradées. Veuillez vider le dossier OUT avant les prochains traitements.")
    elif nb_fichiers_out >= (LIMITE_MAX * 0.8): # À partir de 400 fichiers
        st.warning(f" **Volume élevé ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** Pensez à vider le dossier prochainement pour maintenir une fluidité maximale dans le service.")
    else:
        st.success(f" **Système optimal ({nb_fichiers_out} / {LIMITE_MAX} fichiers).** La lecture des données et la génération des graphiques sont instantanées.")
#===========================================================================================================