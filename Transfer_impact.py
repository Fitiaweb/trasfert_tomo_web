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
Get (relative) sinogram out of RT-PLAN 
--------------------------------------------------------------------------------------------------"""
def get_sinogram(plan):
    
    NCP = plan.BeamSequence[0].NumberOfControlPoints
    sinogram = np.zeros((NCP,64))
    
    """import plan sinogram value at each CP (use of try and except because some CP are empty)"""    
    cp_sequence = plan.BeamSequence[0].ControlPointSequence
    
    for cp in range(NCP):
        try : 
                       
            """(300d,10a7) is the tag where the leaf openings commands are stored"""
            tmp = cp_sequence[cp][0x300d,0x10a7].value
    
            tmp = tmp.decode('utf-8')
            tmp = tmp.split('\\')
            
            """convert the string into float array""" 
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
    return fldpath

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
    
    cond1 = LOT_sino<(maxLOT-1)  #exclude LOT = PT
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
def calc_error_all(): 
    
    fld_path = read_folder() 
                
    plan_list = read_plan(fld_path)
    
    errors_all = []
    
    for i in range(len(plan_list)):
        
        plan = dcm.dcmread(plan_list[i])
        
        tmp = plan_list[i].split('\\')
        
        file_name = tmp[1]
                
        sinogram = get_sinogram(plan)
        info = general_info(plan)
        delivery = delivery_info(plan) 
        
        data = get_error_shift(sinogram,delivery)
        
        undisc_LCT = data[1]
        
        error_plan = data[0]
        
        parts = info["patient_name"].split("^")
        surname = parts[0]
        
        name_id =  str(info["patient_id"]) + ' ' + surname
        
        plan_name = file_name
        
        errors_all.append([plan_name, name_id, undisc_LCT, error_plan])

    return errors_all



"""--------------------------------------------------------------------------------------------------
Create scrollable table with tkinter to plot the supplemantary dose estimations 
--------------------------------------------------------------------------------------------------"""
def create_gui(root, data, columns):
    tree = ttk.Treeview(root, columns=columns, show='headings')
    
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor='center') 
    
    for row in data:
        tree.insert('', 'end', values=row)
    
    tree.grid(row=0, column=0, sticky=tk.NSEW) 

    scrollbar_y = Scrollbar(root, orient=tk.VERTICAL, command=tree.yview)
    scrollbar_y.grid(row=0, column=1, sticky=tk.NS)  
    tree.configure(yscrollcommand=scrollbar_y.set)

    scrollbar_x = Scrollbar(root, orient=tk.HORIZONTAL, command=tree.xview)
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

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

fig = create_gui(root, plot_lines, plot_cols)



""" xlsx file writing with openpyxl"""
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Supplementary dose results" 

ws.append(plot_cols)

for i in range(1,5):
    cell = ws.cell(row=1,column=i)
    cell.font = Font(bold=True)
    cell.alignment = Alignment(horizontal='center', vertical='center')
    

for xcel_rows in data:
    ws.append(xcel_rows)
    
for col in ws.columns:
    max_length = 0
    column_cells = list(col)  # Convertit la colonne en liste
    if len(column_cells) < 3:  # Vérifie si la colonne a au moins 3 cellules
        continue
    
    column = column_cells[2].column_letter
    for cell in column_cells:
        try:
            if len(str(cell.value)) > max_length:
                max_length = len(cell.value)
        except:
            pass
    adjusted_width = (max_length + 2)
    ws.column_dimensions[column].width = adjusted_width

    
out_fld = read_folder_out()
out_file_name = get_out_filename()
    
output_filepath = os.path.join(out_fld,out_file_name)

wb.save(output_filepath)


root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()




























