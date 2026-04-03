<<<<<<< HEAD
#  AVC Risk Predictor — Stroke Prediction with Machine Learning

> **Projet Data Science** — Analyse prédictive des Accidents Vasculaires Cérébraux  
> Développé dans le cadre du programme **Orange Digital Center / Sonatel Académie** · Promotion 7 · Dakar, Sénégal 🇸🇳

---

## Contexte

Avril est le **Mois de sensibilisation aux AVC** au Sénégal.  
L'AVC (Accident Vasculaire Cérébral) touche **1 personne sur 4** dans le monde.  
En tant que Data Analyst en formation, j'ai voulu aller au-delà de la sensibilisation :  
**analyser les données, comprendre les facteurs de risque, et rendre ces insights accessibles à tous.**

---

## Objectifs du projet

- Identifier les **facteurs de risque** les plus critiques d'un AVC
- Construire un **modèle de Machine Learning** capable de prédire le risque
- Développer une **application interactive** (Streamlit) pour permettre à chacun de tester son profil

---

## Dataset

| Propriété | Valeur |
|-----------|--------|
| Source | [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |
| Patients | 4 909 après nettoyage |
| Variables | 11 colonnes (âge, hypertension, maladie cardiaque, tabac...) |
| Variable cible | `stroke` (0 = pas d'AVC · 1 = AVC) |
| Déséquilibre | ~5% de cas positifs → traité avec `class_weight='balanced'` |

### Variables utilisées

| Variable | Description |
|----------|-------------|
| `age` | Âge du patient |
| `hypertension` | 0 = Non · 1 = Oui |
| `heart_disease` | 0 = Non · 1 = Oui |
| `avg_glucose_level` | Glycémie moyenne (mg/dL) |
| `bmi` | Indice de Masse Corporelle |
| `gender` | Male / Female |
| `ever_married` | Marié(e) ou non |
| `smoking_status` | Statut tabagique |

---

## Insights clés

```
🔴 Le risque d'AVC augmente fortement après 50 ans
🔴 L'hypertension multiplie le risque par ~3x
🔴 La maladie cardiaque multiplie le risque par ~4x
🔴 Les fumeurs sont significativement plus exposés
```

---

##  Visualisations

Le projet génère **4 graphiques** :

| Fichier | Contenu |
|---------|---------|
| `graph1_avc_age.png` | Distribution des AVC par tranche d'âge |
| `graph2_avc_cardio.png` | Impact de l'hypertension et maladie cardiaque |
| `graph3_avc_tabac.png` | Risque selon le statut tabagique |
| `graph4_ml_importance.png` | Importance des variables dans le modèle ML |

---

##  Modèle ML

| Paramètre | Valeur |
|-----------|--------|
| Algorithme | Logistic Regression |
| Librairie | Scikit-learn |
| Accuracy | **74%** |
| Stratégie déséquilibre | `class_weight='balanced'` |
| Train/Test split | 80% / 20% (stratifié) |

```
              precision    recall    f1-score
   Sans AVC      0.98       0.74       0.85
   Avec AVC      0.10       0.64       0.17
   accuracy                  0.74
```

>  Le recall de **64%** sur les AVC positifs est l'indicateur le plus important ici :  
> le modèle détecte correctement 64% des cas à risque malgré un fort déséquilibre des classes.

---

##  Lancer le projet

### 1. Cloner le repo

```bash
git clone https://github.com/TON_USERNAME/avc-risk-predictor.git
cd avc-risk-predictor
```

### 2. Installer les dépendances

```bash
pip install pandas matplotlib scikit-learn numpy streamlit
```

### 3. Générer les graphiques + entraîner le modèle

```bash
python avc_complet.py
```

### 4. Lancer l'application interactive

```bash
streamlit run app.py
```

---

##  Application Streamlit

L'application permet à n'importe qui de :

1. Renseigner ses données personnelles (âge, IMC, glycémie...)
2. Indiquer ses antécédents médicaux
3. Obtenir une **estimation de son risque d'AVC** avec un niveau de confiance

| Niveau | Seuil |
|--------|-------|
| 🟢 Risque faible | < 25% |
| 🟡 Risque modéré | 25% – 50% |
| 🔴 Risque élevé | > 50% |

>  **Avertissement** : Cette application est un projet éducatif.  
> Elle ne remplace en aucun cas un diagnostic médical professionnel.

---

##  Structure du projet

```
avc-risk-predictor/
│
├── healthcare-dataset-stroke-data.csv   # Dataset source
├── avc_complet.py                       # Script analyse + graphiques + ML
├── app.py                               # Application Streamlit
├── graph1_avc_age.png                   # Graphique 1
├── graph2_avc_cardio.png                # Graphique 2
├── graph3_avc_tabac.png                 # Graphique 3
├── graph4_ml_importance.png             # Graphique 4 (importance ML)
└── README.md
```

---

##  Stack technique

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-11557c)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit&logoColor=white)

---

##  Signes d'alerte AVC

> Si vous observez l'un de ces signes, **appelez les secours immédiatement** :

-  Visage qui tombe d'un côté
-  Faiblesse soudaine d'un bras ou d'une jambe
-  Difficulté à parler ou à comprendre
-  Maux de tête brutaux et intenses
-  Troubles de la vision soudains

📞 **Numéro vert Sénégal : 800 00 50 50**

---

## 👤 Auteur

**Serigne** · Data Analyst & Data Engineer Junior  
🎓 Orange Digital Center / Sonatel Académie · Promotion 7  
📍 Dakar, Sénégal  

[LinkedIn](www.linkedin.com/in/serigne-babacar-kane-6b9759206)
[GitHub](https://github.com/Goorgui-5)

---

*Projet réalisé en avril 2026 — Mois de sensibilisation aux AVC *
=======
# Projet_AVC
Outil de prédiction du risque d'AVC basé sur le Machine Learning — Logistic Regression entraînée sur 4 909 patients. Renseignez vos données cliniques et obtenez une estimation personnalisée en quelques secondes.
>>>>>>> 6ffa186af33619d4ee00d3469e2dcbcfcb54d8b7
