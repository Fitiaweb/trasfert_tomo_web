# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:40:39 2024

@author: seguret_j
"""

import tkinter as tk
from tkinter import filedialog 
from tkinter import ttk, Scrollbar, simpledialog

import pydicom as dcm

import os
import numpy as np

import openpyxl
from openpyxl.styles import Font, Alignment

import sys


"""--------------------------------------------------------------------------------------------------
Get (relative) sinogram out of RT-PLAN
--------------------------------------------------------------------------------------------------"""
def get_sinogram(plan):
    
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    sinogram = np.zeros((NCP,64)) 
    
    cp_sequence = plan.BeamSequence[0].ControlPointSequence 

    for cp in range(NCP): 
        try : 
            tmp = cp_sequence[cp][0x300d,0x10a7].value 
            tmp = tmp.decode('utf-8')
            tmp = tmp.split('\\')
            sinogram[cp-1,:] = np.array(tmp,dtype=np.float64)  
        except KeyError:
            continue 
        
    return sinogram 


"""--------------------------------------------------------------------------------------------------
Get general information (patient ID, machine type etc...) out of RT-PLAN 
--------------------------------------------------------------------------------------------------"""
def general_info(plan) : 
    
    plan_info = {}
    plan_info["plan_name"] = plan.RTPlanName
    plan_info["patient_id"] = plan.PatientID
    plan_info["patient_name"] = str(plan.PatientName)
    plan_info["patient_bday"] = plan.PatientBirthDate
    plan_info["patient_sex"] = plan.PatientSex
    plan_info["plan_date"] = plan.RTPlanDate
    plan_info["manufacturer"] = plan.ManufacturerModelName
    
    serial_mapping = {
        "4010012": "Tomo2", 
        "210462": "Tomo4",  
        "4010710": "Tomo7"
    }
    
    try:
        raw_serial = str(plan.DeviceSerialNumber)
        plan_info["machine_nb"] = serial_mapping.get(raw_serial, raw_serial)
    except AttributeError:
        plan_info["machine_nb"] = "Inconnue"

    return plan_info


"""--------------------------------------------------------------------------------------------------
Get delivery information (Gantry period, Nb of rotations, Couch Speed etc...) out of RT-PLAN 
--------------------------------------------------------------------------------------------------"""
def delivery_info(plan) : 

    delivery = {}
    delivery["GP"] = float(plan.BeamSequence[0][0x300d,0x1040].value) 
    delivery["PT"] = (delivery["GP"]/51.0)*1000.0   
    delivery["CS"] = float(plan.BeamSequence[0][0x300d,0x1080].value) 
    delivery["pitch"] = float(plan.BeamSequence[0][0x300d,0x1060].value) 
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    delivery["Nrot"] = (NCP-1)/51    
    delivery["TT"] = delivery["Nrot"]*delivery["GP"] 
    delivery["CT"] = delivery["TT"]*delivery["CS"] 
    delivery["FW"] = round(delivery["CT"]/delivery["Nrot"]/delivery["pitch"]/10.0,1) 
    delivery["TL"] = delivery["CT"] - delivery["FW"]*10.0 

    try:
        delivery["DS"] = float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose)
    except Exception:
        delivery["DS"] = 0.0

    delivery["TTDF"] = (float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose))/delivery["TT"]*100.0 
    
    return delivery


"""--------------------------------------------------------------------------------------------------
Get the folder where RT-PLAN are stored 
--------------------------------------------------------------------------------------------------""" 
def read_folder():
    fldpath = "rp/"
    return fldpath  


"""--------------------------------------------------------------------------------------------------
Get the list of plans with the specified path
--------------------------------------------------------------------------------------------------"""
def read_plan(fldpath):
    path_list = []
    for root, dirs, files in os.walk(fldpath):  
        for file in files:
            if file.startswith("RP") and file.endswith(".dcm"):  
                path_list.append(os.path.join(root, file))  
    return path_list


"""--------------------------------------------------------------------------------------------------
Get the percentage of dose error when a plan is transfered from a machine to another
--------------------------------------------------------------------------------------------------"""
def get_error_shift(sinogram,delivery):

    PT = delivery["PT"] 
    LOT_sino = PT*sinogram
    maxLOT = np.max(LOT_sino)
    total_lot = np.sum(LOT_sino)
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  
    
    thresh = 18 
    undisc_LCT = 0
    
    cond1 = LOT_sino<(maxLOT-1)  
    cond2 = LOT_sino>(PT-thresh) 

    row,col = np.where(cond1 & cond2)
    
    for i in range(len(row)):
        if row[i] < LOT_sino.shape[0]:            
            if (LOT_sino[row[i]+1,col[i]] > (PT-20)):
                undisc_LCT = undisc_LCT+1 
        else : 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row>(-0.5)] 
    filtered_col = col[col>(-0.5)]
    extra_time = 0  
    
    for i in range(len(filtered_row)-1):
        extra_time = extra_time + (PT - LOT_sino[filtered_row[i],filtered_col[i]])  

    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100) 
            

"""--------------------------------------------------------------------------------------------------
Calculate the estimated error for the list of plans in the chosen folder 
--------------------------------------------------------------------------------------------------"""
def calc_error_all(): 
    
    fld_path = read_folder() 
    plan_list = read_plan(fld_path) 
    raw_data = [] 
    
    for i in range(len(plan_list)):
        plan = dcm.dcmread(plan_list[i]) 
        sinogram = get_sinogram(plan) 
        info = general_info(plan) 
        delivery = delivery_info(plan) 
        data = get_error_shift(sinogram,delivery) 
        
        undisc_LCT = data[1] 
        error_plan = data[0] 
            
        parts = info["patient_name"].split("^")  
        surname = parts[0] 
        first_name = parts[1] if len(parts) > 1 else "" 
        
        name_id = f"{first_name} {surname}".strip()
        patient_id = str(info["patient_id"])
        
        try:
            plan_date = str(plan[0x0008, 0x0012].value)
        except KeyError:
            try:
                plan_date = str(plan[0x300A, 0x0006].value)
            except KeyError:
                plan_date = "00000000" 

        try:
            plan_time = str(plan[0x0008, 0x0013].value)
        except KeyError:
            try:
                plan_time = str(plan[0x300A, 0x0007].value)
            except KeyError:
                plan_time = "000000"

        raw_data.append({
            'sort_key': plan_date + plan_time, 
            'plan_date': plan_date,
            'plan_time': plan_time,
            'name_id': name_id,
            'patient_id': patient_id,
            'machine_serial': info["machine_nb"],
            'dose_fraction': delivery["DS"],
            'undisc_LCT': undisc_LCT,
            'error_plan': error_plan
        })

    raw_data.sort(key=lambda x: x['sort_key'])

    patient_display = raw_data[0]['name_id'] if len(raw_data) > 0 else "Patient Inconnu"
    id_display = raw_data[0]['patient_id'] if len(raw_data) > 0 else "ID Inconnu"

    # Calcul de l'erreur globale finale (pour la sécurité des séances max)
    total_cumulative_error = 0.0
    for index, item in enumerate(raw_data):
        if index > 0: 
            total_cumulative_error += item['error_plan']

    errors_all = [] 
    running_cumulative_error = 0.0 # NOUVEAU: Pour le suivi progressif ligne par ligne
    
    for index, item in enumerate(raw_data):
        
        error_plan_val = item['error_plan']
        affichage_uLCT = f"{item['undisc_LCT']:.2f}"
        affichage_dose = f"{item['dose_fraction']:.2f}"

        commentaires = []
        
        if index == 0:
            # Pour l'initiale, on cache les valeurs d'erreur
            affichage_error_plan = "-"
            affichage_cumul = "-"
            affichage_seances = "-"
            commentaire = "Plan de référence"
        else:
            # Ajout progressif à l'erreur cumulée
            running_cumulative_error += error_plan_val
            
            affichage_error_plan = f"{error_plan_val:.2f}"
            affichage_cumul = f"{running_cumulative_error:.2f}" # Affiche la progression
            
            # Le budget utilise toujours l'erreur TOTALE finale, c'est ce qui est important !
            budget_restant = 10.0 - total_cumulative_error
            
            if error_plan_val > 0:
                if budget_restant > 0:
                    seances_restantes = int(budget_restant / error_plan_val)
                    affichage_seances = f"{seances_restantes}"
                else:
                    affichage_seances = "0"
            else:
                affichage_seances = "Illimité"

            # Alertes
            if error_plan_val > 1.5:
                commentaires.append("Dose/séance > 1.5%")
            if total_cumulative_error >= 10.0:
                commentaires.append("Seuil 10% DÉPASSÉ !")
                
            commentaire = " | ".join(commentaires)
        
        machine_serial = item['machine_serial']
        nom_patient = item['name_id']
        
        date_brute = item['plan_date']
        if len(date_brute) == 8:
            date_formatee = f"{date_brute[6:8]}/{date_brute[4:6]}/{date_brute[0:4]}"
        else:
            date_formatee = date_brute 

        time_brute = item['plan_time']
        if len(time_brute) >= 6:
            time_formatee = f"{time_brute[0:2]}:{time_brute[2:4]}:{time_brute[4:6]}"
        else:
            time_formatee = time_brute

        datetime_formatee = f"{date_formatee} à {time_formatee}"

        if index == 0:
            date_affichage = f"{datetime_formatee} (Initiale)"
        else:
            date_affichage = datetime_formatee

        errors_all.append([date_affichage, nom_patient, machine_serial, affichage_dose, affichage_uLCT, affichage_error_plan, affichage_cumul, affichage_seances, commentaire])

    return patient_display, id_display, errors_all


"""--------------------------------------------------------------------------------------------------
Create scrollable table with tkinter to plot the supplemantary dose estimations 
--------------------------------------------------------------------------------------------------"""
def create_gui(root, data, columns, patient_name, patient_id): 
    
    lbl_name = tk.Label(root, text=f"Patient : {patient_name}", font=('Arial', 14, 'bold'))
    lbl_name.grid(row=0, column=0, sticky=tk.NW, padx=20, pady=(20, 0))
    
    lbl_id = tk.Label(root, text=f"ID : {patient_id}", font=('Arial', 14, 'bold'))
    lbl_id.grid(row=0, column=0, sticky=tk.NE, padx=20, pady=(20, 0))

    style = ttk.Style()
    style.theme_use("clam") 
    
    style.configure("Treeview.Heading", font=('Arial', 10, 'bold'), background="#4a90e2", foreground="white")   
    style.configure("Treeview", font=('Arial', 10), rowheight=30) 
    
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.grid(row=1, column=0, sticky=tk.NSEW)

    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)

    tree = ttk.Treeview(main_frame, columns=columns, show='headings')

    for i, col in enumerate(columns):
        tree.heading(col, text=col)
        # Ajustement des largeurs
        if "Date" in col:
            width = 180
        elif "Nom" in col:
            width = 150
        elif "Machine" in col:
            width = 90
        elif "Dose" in col:
            width = 130
        elif "uLCT" in col or "Estimated" in col or "Erreur cumulée" in col:
            width = 160 
        elif "restantes" in col:
            width = 140
        elif "Commentaire" in col:
            width = 250
        else:
            width = 100
            
        tree.column(col, width=width, anchor='center') 
    
    tree.tag_configure('pair', background="#f9f9f9")    
    tree.tag_configure('impair', background="#ffffff")
    tree.tag_configure('initiale', background="#d1e7dd", font=('Arial', 10, 'bold')) 
    tree.tag_configure('alerte', background="#ffebee", foreground="#d32f2f", font=('Arial', 10, 'bold'))

    for index, row in enumerate(data):
        
        is_alert = False
        
        # Pour déclencher l'alerte, on regarde si la dose > 1.5 ou si le commentaire dit "DÉPASSÉ"
        if index > 0:
            try:
                est_dose = float(row[5])
                if est_dose > 1.5:
                    is_alert = True
            except ValueError:
                pass
            
            # Si le texte "DÉPASSÉ" apparaît dans la colonne Commentaire, on met en rouge
            if "DÉPASSÉ" in row[8]:
                is_alert = True

        if index == 0:
            tag = 'initiale' 
        else:
            if is_alert:
                tag = 'alerte'
            else:
                tag = 'pair' if index % 2 == 0 else 'impair'
            
        tree.insert('', 'end', values=row, tags=(tag,))
    
    tree.grid(row=0, column=0, sticky=tk.NSEW) 

    scrollbar_y = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar_y.grid(row=0, column=1, sticky=tk.NS)  
    tree.configure(yscrollcommand=scrollbar_y.set)

    scrollbar_x = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=tree.xview)
    scrollbar_x.grid(row=1, column=0, sticky=tk.EW) 
    tree.configure(xscrollcommand=scrollbar_x.set)

    def on_mousewheel(event):
        if event.delta > 0:
            tree.yview_scroll(-1, 'units')
        elif event.delta < 0:
            tree.yview_scroll(1, 'units')
    tree.bind('<MouseWheel>', on_mousewheel)

    return tree

"""--------------------------------------------------------------------------------------------------
Function that stops the tkinter interactive window job when closing the window 
--------------------------------------------------------------------------------------------------"""
def on_closing():
    root.destroy()
    sys.exit()
    
"""--------------------------------------------------------------------------------------------------
Get the folder for output 
--------------------------------------------------------------------------------------------------""" 
def read_folder_out():
    fldpath = "rp/"
    return fldpath

"""--------------------------------------------------------------------------------------------------
Get output file name from user with tkinter 
--------------------------------------------------------------------------------------------------"""
def get_out_filename():
    user_input = "output_data.xlsx"
    return user_input


"""--------------------------------------------------------------------------------------------------
************************************* MAIN       **********************************************
--------------------------------------------------------------------------------------------------"""

patient_name, patient_id, data = calc_error_all()

plot_cols = ["Date Heure", "Nom", "Machine", "Dose / Fraction", "uLCT (%)", "Estimated dose/séance (%)", "Erreur cumulée (%)", "Séances max", "Commentaire"]

plot_lines = []

for sub_list in data: 
    plot_lines.append(sub_list)
    
root = tk.Tk()
root.title("Estimation des erreurs de dose et Sécurité")

root.geometry("1650x600") 

root.grid_rowconfigure(0, weight=0) 
root.grid_rowconfigure(1, weight=1) 
root.grid_columnconfigure(0, weight=1) 

fig = create_gui(root, plot_lines, plot_cols, patient_name, patient_id) 

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()