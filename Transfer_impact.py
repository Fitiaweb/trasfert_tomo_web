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




"""--------------------------------------------------------------------------------------------------ee
Get (relative) sinogram out of RT-PLAN , ceci
--------------------------------------------------------------------------------------------------"""
def get_sinogram(plan):
    
    NCP = plan.BeamSequence[0].NumberOfControlPoints  #nb of control point 
    sinogram = np.zeros((NCP,64)) #64 is the number of leaf pairs for our machines, it can be changed if needed for other machines (but it is not expected to be different) 
                                   # 64 lames Ouvert ou Fermé
    
    """import plan sinogram value at each CP (use of try and except because some CP are empty)"""    
    cp_sequence = plan.BeamSequence[0].ControlPointSequence #sequance of control point ( pour 20 rotation : 51 *20 + 1 = 1021 CP)
                                                            # 51 pour nb de projection | 20 pour nb de rotation | +1 index 0 

    
    for cp in range(NCP): #python gère tout seul l'incrémentation de cp 
        try : 
                       
            """(300d,10a7) is the tag where the leaf openings commands are stored"""
            tmp = cp_sequence[cp][0x300d,0x10a7].value #obtient la position des lames 
    
            tmp = tmp.decode('utf-8')# devient une chaine 
            tmp = tmp.split('\\')# devient une liste de chaines 
            
            """convert the string into float array""" 
            sinogram[cp-1,:] = np.array(tmp,dtype=np.float64)  #???????????????????? convertion en float mais pk cp-1  (: -> tout les elements sur cet axe)
            
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
    plan_info["machine_nb"] = plan.DeviceSerialNumber
    
    return plan_info



"""--------------------------------------------------------------------------------------------------
Get delivery information (Gantry period, Nb of rotations, Couch Speed etc...) out of RT-PLAN 
--------------------------------------------------------------------------------------------------"""
def delivery_info(plan) : 

    delivery = {}
    delivery["GP"] = float(plan.BeamSequence[0][0x300d,0x1040].value) #Gantry Period (s)
    delivery["PT"] = (delivery["GP"]/51.0)*1000.0   #Projection Time (ms)
    delivery["CS"] = float(plan.BeamSequence[0][0x300d,0x1080].value) #Couch Speed (mm/s)
    delivery["pitch"] = float(plan.BeamSequence[0][0x300d,0x1060].value) #Pitch
    NCP = plan.BeamSequence[0].NumberOfControlPoints  
    delivery["Nrot"] = (NCP-1)/51    #PNumber of gantry rotations
    delivery["TT"] = delivery["Nrot"]*delivery["GP"] #Treatment Time (s)
    delivery["CT"] = delivery["TT"]*delivery["CS"] #Couch Translation (mm)
    delivery["FW"] = round(delivery["CT"]/delivery["Nrot"]/delivery["pitch"]/10.0,1) #Field Width (cm)
    delivery["TL"] = delivery["CT"] - delivery["FW"]*10.0 #Target Length (mm)
    delivery["TTDF"] = (float(plan.FractionGroupSequence[0].ReferencedBeamSequence[0].BeamDose))/delivery["TT"]*100.0 #Dose over time (cGy/s)
    
    return delivery



"""--------------------------------------------------------------------------------------------------
Get the folder where RT-PLAN are stored 
--------------------------------------------------------------------------------------------------""" 
def read_folder():

    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) 

    fldpath = filedialog.askdirectory(title="Sélectionner le dossier contenant le(s) RT-PLAN")
    """
    fldpath = "rp/"
    return fldpath  # chercher le dossier rp et l'ouvrir en arriere plan sans l'afficher pour l'utilisateur 

"""--------------------------------------------------------------------------------------------------
Get the list of plans with the specified path
--------------------------------------------------------------------------------------------------"""
"""def read_plan(fldpath):

    plan_list = os.listdir(fldpath)

    path_list = []

    for i in range(len(plan_list)):
        path_list.append(os.path.join(fldpath, plan_list[i]))

    return path_list """



def read_plan(fldpath):
    path_list = []

    for root, dirs, files in os.walk(fldpath):  # Parcours récursif
        for file in files:
            if file.startswith("RP") and file.endswith(".dcm"):  # Filtrer les fichiers
                path_list.append(os.path.join(root, file))  # Ajouter le chemin complet

    return path_list


"""--------------------------------------------------------------------------------------------------
Get the percentage of dose error when a plan is transfered from a machine to another
--------------------------------------------------------------------------------------------------"""
def get_error_shift(sinogram,delivery):

    PT = delivery["PT"] 
    LOT_sino = PT*sinogram
    maxLOT = np.max(LOT_sino)
    total_lot = np.sum(LOT_sino)
    open_leaves_LOT = LOT_sino[np.nonzero(LOT_sino)]  #size of open leaves corresponds to N_open (see publication)
    
    thresh = 18 #max leaf transitionj time is taken as 18 ms accuray
    undisc_LCT = 0
    
    cond1 = LOT_sino<(maxLOT-1)  #exclude LOT = PT      ???pk pas juste LOT_sino < PT

    cond2 = LOT_sino>(PT-thresh) #LOT is a short LCT 

    row,col = np.where(cond1 & cond2)
    
    for i in range(len(row)):
        
        if row[i] < LOT_sino.shape[0]:           
            if (LOT_sino[row[i]+1,col[i]] > (PT-20)):
                undisc_LCT = undisc_LCT+1 
                #iterate if current LOT is in the range PT-18ms AND next or previous LOT (for a given leaf) is in the range PT-20ms
                #use of PT-20 ms because mean latency offset of our machines is 2ms
                
        else : 
            row[i] = -1 
            col[i] = -1 
            
    filtered_row = row[row>(-0.5)] #only get the sinograms entries corresponding to short LCT
    filtered_col = col[col>(-0.5)]
    extra_time = 0  
    
    for i in range(len(filtered_row)-1):
        extra_time = extra_time + (PT - LOT_sino[filtered_row[i],filtered_col[i]])  # add additional time to get the sum of it

    return ((extra_time/total_lot)*100), ((undisc_LCT/(len(open_leaves_LOT)))*100) # return the percentage of this extra time compared to sum of all LOTs
            

"""--------------------------------------------------------------------------------------------------
Calculate the estimated error for the list of plans in the chosen folder 
--------------------------------------------------------------------------------------------------"""
def calc_error_all(): # équivalent d'un pré main 
    
    fld_path = read_folder() #recupérer le chemin du dossier contenant les plans
                
    plan_list = read_plan(fld_path) #plan_list contient la liste des chemins d'accès de tous les plans du dossier choisi
    
    errors_all = [] 
    
    for i in range(len(plan_list)):
        
        plan = dcm.dcmread(plan_list[i]) #convertir le plan en objet pydicom pour pouvoir accéder à ses données facilement
        
        tmp = plan_list[i].split('\\') 
        
        file_name = tmp[1] 
                
        sinogram = get_sinogram(plan) #appeler la fonction qui récupère le sinogramme du plan pour pouvoir calculer l'erreur de dose ensuite
        info = general_info(plan) #appeler la fonction qui récupère les informations générales du plan pour pouvoir les afficher ensuite dans le tableau
        delivery = delivery_info(plan) #appeler la fonction qui récupère les informations de livraison du plan pour pouvoir les afficher ensuite dans le tableau
        
        data = get_error_shift(sinogram,delivery) #appeler la fonction qui calcule l'erreur de dose pour ce plan en utilisant le sinogramme et les informations de livraison du plan
        
        undisc_LCT = data[1] #donnée 1 pourcentage de LCT non détecté 
        
        error_plan = data[0] #donnée 0 pourcentage d'erreur de dose estimé pour ce plan
            
        parts = info["patient_name"].split("^") #donner le nom du patient 
        surname = parts[0] 
        
        name_id =  str(info["patient_id"]) + ' ' + surname
        
        plan_name = file_name

        ##
        if undisc_LCT == 0.0:
            affichage_uLCT = "0.0 ( image initiale )"
        else:
            affichage_uLCT = undisc_LCT
        ##

        errors_all.append([plan_name, name_id, affichage_uLCT, error_plan])

    return errors_all



"""--------------------------------------------------------------------------------------------------
Create scrollable table with tkinter to plot the supplemantary dose estimations 
--------------------------------------------------------------------------------------------------"""
def create_gui(root, data, columns): 
    ## =========================== modif 19/05/2026



    style = ttk.Style()
    style.theme_use("clam") # Thème plus moderne 
    

    style.configure("Treeview.Heading", font=('Arial', 10, 'bold'), background="#4a90e2", foreground="white")   #bg #4a90e2
    

    style.configure("Treeview", font=('Arial', 10), rowheight=30) 
    

    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.grid(row=1, column=0, sticky=tk.NSEW)

    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    main_frame.grid_rowconfigure(0, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)


    tree = ttk.Treeview(main_frame, columns=columns, show='headings')



    ## ============================

    for i, col in enumerate(columns):
        tree.heading(col, text=col)

        width = 250 if i < 2 else 150
        tree.column(col, width=width, anchor='center') 
    

    tree.tag_configure('pair', background="#f9f9f9")    # ligne pair et impair tableau 
    tree.tag_configure('impair', background="#ffffff")
    

    for index, row in enumerate(data):
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

    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True) 

    fldpath = filedialog.askdirectory(title="Sélectionner le dossier pour stocker le fichier résultat")
    """
    fldpath = "rp/"
    return fldpath


"""--------------------------------------------------------------------------------------------------
Get output file name from user with tkinter 
--------------------------------------------------------------------------------------------------"""
def get_out_filename():

    """
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 

    # ask user to input 
    user_input = simpledialog.askstring(title="Input", prompt="Entrez le nom du fichier xlsx de sortie:")
    
    user_input = user_input + ".xlsx"
    """
    user_input = "output_data.xlsx"
    return user_input

        
"""--------------------------------------------------------------------------------------------------
*************************************       MAIN       **********************************************
--------------------------------------------------------------------------------------------------"""

data = calc_error_all()

plot_cols = ["ID and Plan name", "Name", "uLCT (%)", "Estimated additional dose (%)"] 
plot_lines = []

for sub_list in data: 
    plot_lines.append(sub_list)
    
root = tk.Tk()
root.title("Estimation des erreurs de dose")

##============modif 19/05/2026
root.geometry("1400x600")
##============

root.grid_rowconfigure(0, weight=1) 
root.grid_columnconfigure(0, weight=1) 

fig = create_gui(root, plot_lines, plot_cols) 

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
