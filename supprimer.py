import sqlite3

# Remplace par l'ID du patient que tu veux effacer
ID_A_SUPPRIMER = "202605485" 

conn = sqlite3.connect("tomo_database.db")
cursor = conn.cursor()

# 1. On supprime d'abord toutes les séances de ce patient
cursor.execute("DELETE FROM SEANCES WHERE Patient_ID = ?", (ID_A_SUPPRIMER,))
seances_supprimees = cursor.rowcount

# 2. On supprime ensuite le patient lui-même de la liste
cursor.execute("DELETE FROM PATIENTS WHERE Patient_ID = ?", (ID_A_SUPPRIMER,))
patient_supprime = cursor.rowcount

# 3. On sauvegarde !
conn.commit()
conn.close()

print(f"Nettoyage terminé : {patient_supprime} patient et {seances_supprimees} séance(s) supprimés.")