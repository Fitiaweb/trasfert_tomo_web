import sqlite3

# Le nom de ton fichier de base de données
DB_NAME = "tomo_database.db"

def initialiser_base_de_donnees():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 1. Création de la table PATIENTS
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS PATIENTS (
        Patient_ID TEXT PRIMARY KEY,
        Nom TEXT,
        Prenom TEXT
    )
    ''')

    # 2. Création de la table PRESCRIPTIONS (Complète !)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS PRESCRIPTIONS (
        Prescription_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Patient_ID TEXT,
        DICOM_SOP_UID TEXT UNIQUE, -- Filtre anti-doublon
        Date_Planification TEXT,
        Budget_Total_Gy REAL,
        
        -- Les paramètres physiques (La technique du plan)
        Gantry_Period REAL,          
        Projection_Time REAL,        
        Couch_Speed REAL,            
        Pitch REAL,                  
        Nombre_Rotations REAL,       
        Treatment_Time REAL,         
        
        FOREIGN KEY (Patient_ID) REFERENCES PATIENTS(Patient_ID)
    )
    ''')

    # 3. Création de la table SEANCES
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS SEANCES (
        Seance_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        Patient_ID TEXT,
        Prescription_ID INTEGER,
        DICOM_SOP_UID TEXT UNIQUE, -- Filtre anti-doublon
        Date_Heure_Seance TEXT,
        Machine_Utilisee TEXT,
        Dose_Delivree_Gy REAL,
        Profil_Erreur_JSON TEXT,
        
        FOREIGN KEY (Patient_ID) REFERENCES PATIENTS(Patient_ID),
        FOREIGN KEY (Prescription_ID) REFERENCES PRESCRIPTIONS(Prescription_ID)
    )
    ''')

    # On valide et on ferme
    conn.commit()
    conn.close()

initialiser_base_de_donnees()