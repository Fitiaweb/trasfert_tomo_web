import sqlite3

# On ouvre le coffre-fort
conn = sqlite3.connect("tomo_database.db")
cursor = conn.cursor()

print("==========================================================")
print("          VÉRIFICATION DE LA BASE DE DONNÉES COMPLÈTE     ")
print("==========================================================\n")

# --- 1. LECTURE DE LA TABLE PATIENTS ---
print("--- TABLE 'PATIENTS' ---")
cursor.execute("SELECT * FROM PATIENTS;")
patients = cursor.fetchall()

if len(patients) == 0:
    print("La table PATIENTS est vide.")
else:
    # Récupère le nom des colonnes
    noms_colonnes_pat = [description[0] for description in cursor.description]
    print(f"Colonnes : {noms_colonnes_pat}")
    
    for p in patients:
        print(p)

print("\n" + "=" * 60 + "\n")

# --- 2. LECTURE DE LA TABLE SEANCES ---
print("--- TABLE 'SEANCES' (Toutes les colonnes) ---")
try:
    # L'étoile (*) veut dire "Prends TOUTES les colonnes sans exception"
    cursor.execute("SELECT * FROM SEANCES;")
    seances = cursor.fetchall()

    if len(seances) == 0:
        print("La table SEANCES est vide.")
    else:
        # Récupère le nom de toutes les colonnes dynamiquement
        noms_colonnes = [description[0] for description in cursor.description]
        print(f"Colonnes : {noms_colonnes}\n")
        
        for s in seances:
            # Comme le Profil_JSON (le dernier élément) contient 51 valeurs, 
            # on l'affiche en entier mais je le mets à la fin pour ne pas casser la lecture
            print(f"ID: {s[0]} | Patient_ID: {s[1]} | UID: {s[2]}")
            print(f"Date_tri: {s[3]} | Date_Affichee: {s[4]} | Machine: {s[5]}")
            print(f"Dose: {s[6]} Gy | Frac: {s[7]} | uLCT: {s[8]}% | Erreur: {s[9]}%")
            # On coupe l'affichage du JSON à 50 caractères pour que ton terminal reste lisible, 
            # mais tout est bien dans la base !
            print(f"Profil JSON (aperçu) : {str(s[10])[:50]}...\n")
            print("-" * 40)
            
except sqlite3.OperationalError as e:
    print(f"Erreur : {e}")

conn.close()