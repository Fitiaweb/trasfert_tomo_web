# 📊 Tableau de Bord : Impact des Transferts TomoTherapy

<p align="justify">
Cette application web Python, propulsée par Streamlit, permet d'évaluer la dose supplémentaire induite par les temps de fermeture des lames courts (<i>short LCTs - Leaf Closing Times</i>) générés lors du transfert d'un plan de traitement TomoTherapy entre deux machines.
</p>

<p align="justify">
Ce phénomène affecte des localisations spécifiques, avec des erreurs de dose potentielles pouvant atteindre 2,5 à 3 % si la distribution initiale présente un pourcentage élevé de temps d'ouverture des lames (LOT) strictement égal au temps de projection (PT). En attendant un correctif constructeur (Accuray) pour écarter ces LCT courts lors des transferts, ce tableau de bord sert d'<b>outil d'assurance qualité (QA) robuste</b> en calculant la quantité de LCT non écartés, la somme de leurs temps d'ouverture supplémentaires, et la dose additionnelle délivrée.
</p>

---

## ✨ Fonctionnalités Principales

* **🏥 Interface Clinique Épurée :** Une interface web moderne conçue pour un accès clinique rapide, incluant une visualisation de l'historique patient et un suivi de la dose cumulée.
* **📂 Gestion Automatisée des Fichiers :** Système d'ingestion intuitif. Le script surveille de manière récursive le répertoire d'entrée, traite les fichiers DICOM et les archive automatiquement.
* **🗄️ Base de Données Intégrée :** Utilisation de SQLite (`tomo_database.db`) pour un historique automatisé et pérenne des dossiers patients et de l'évolution des séances.
* **🔮 Simulateur de Fin de Traitement :** Algorithme prédictif strict permettant d'anticiper la répartition des séances restantes sans dépasser le budget de dose toléré.
* **📄 Rapports Automatiques :** Génération instantanée de bilans cliniques au format PDF.

---

## 🏗️ Architecture du Répertoire

L'application gère automatiquement deux répertoires essentiels à la racine du projet pour fonctionner :

* `IN/` : Dossier de dépôt. Placez ici vos nouveaux fichiers DICOM RT-PLAN (`RP*.dcm` ou `RTPLAN*.dcm`).
* `ARCHIVES/` : Dossier de stockage. Les fichiers traités y sont automatiquement déplacés pour maintenir l'intégrité de la base de données et la propreté du dossier.

---

## 🚀 Guide de Déploiement

Pour déployer ce tableau de bord sur une machine locale ou un poste dédié dans le service, suivez ces étapes :

### 1. Prérequis
Assurez-vous que **Python (3.9 ou supérieur)** est installé sur la machine. 
*(⚠️ N'oubliez pas de cocher la case "Add Python to PATH" lors de l'installation Windows).*

### 2. Installation de l'environnement
Ouvrez une invite de commande (`cmd` ou PowerShell) dans le dossier du projet et exécutez les commandes suivantes :

```bash
# Création de l'environnement virtuel
python -m venv .venv

# Activation de l'environnement (Windows)
.\.venv\Scripts\activate

# Installation des dépendances requises
pip install streamlit pandas pydicom matplotlib plotly fpdf
```

### 3. Lancement de l'Application
Une fois l'environnement configuré et activé, lancez le serveur Streamlit avec la commande suivante :

```bash
streamlit run Transfer_impact.py
```
*Note : Un script `Launch_Dashboard.bat` peut être configuré pour automatiser ce lancement en un double-clic pour les utilisateurs cliniques.*

---

## ⚙️ Compatibilité

Ce code est exclusivement compatible avec les fichiers **RT-PLAN** générés par le TPS **Accuray Precision** (Accuray Inc., Madison, Wisconsin, USA).

---

## 👨‍💻 Contexte & Auteur

* **Auteur :** Témoë Delsol
* **Contexte :** Projet développé au sein du département de Physique Médicale de l'IUCT Oncopole (Toulouse) dans le cadre du cursus de Génie Biomédical.
* **Remerciements :** Ce projet est dérivé des travaux initiaux et du dépôt GitHub de Julien Seguret (`tomotherapy-transfer-impact`).
