import sqlite3

# 1. On ouvre le coffre-fort
conn = sqlite3.connect("tomo_database.db")
cursor = conn.cursor()

# 2. On demande à SQLite la liste de toutes les tables qui existent
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("--- TABLES DANS LA BASE ---")
for table in tables:
    print("-", table[0])

# 3. (Optionnel) Si on veut voir ce qu'il y a dans la table PRESCRIPTIONS
cursor.execute("SELECT * FROM PRESCRIPTIONS;")
lignes = cursor.fetchall()

print("\n--- CONTENU DE LA TABLE PRESCRIPTIONS ---")
if len(lignes) == 0:
    print("La table est vide pour l'instant !")
else:
    for ligne in lignes:
        print(ligne)

conn.close()