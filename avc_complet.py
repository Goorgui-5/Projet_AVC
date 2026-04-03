import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
#   CHARGEMENT
# ═══════════════════════════════════════════════════════════
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(" Dataset chargé :", df.shape)

# ═══════════════════════════════════════════════════════════
# 🧹  NETTOYAGE
# ═══════════════════════════════════════════════════════════

# 1. Voir les colonnes
print("\n Colonnes :", list(df.columns))

# 2. Supprimer la colonne id
df = df.drop("id", axis=1)

# 3. Vérifier les valeurs manquantes avant
print("\n Valeurs manquantes avant nettoyage :")
print(df.isnull().sum())

# 4. Supprimer les lignes avec valeurs manquantes
df = df.dropna()

# 5. Vérifier après
print("\n Valeurs manquantes après nettoyage :")
print(df.isnull().sum())

# 6. Vérifier les types
print()
df.info()

# 7. Supprimer la ligne "Other" dans gender (trop rare)
df = df[df['gender'] != 'Other']

print(f"\n Dataset final prêt : {df.shape[0]} lignes | {df.shape[1]} colonnes")
print(f"   AVC positifs : {df['stroke'].sum()} ({df['stroke'].mean()*100:.1f}%)\n")

# ═══════════════════════════════════════════════════════════
#   STYLE GLOBAL
# ═══════════════════════════════════════════════════════════
BLUE  = "#2563EB"
RED   = "#EF4444"
GRAY  = "#94A3B8"
BG    = "#F8FAFC"
DARK  = "#1E293B"

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.facecolor"    : BG,
    "figure.facecolor"  : "white",
})

# ═══════════════════════════════════════════════════════════
#   GRAPHIQUE 1 — AVC vs ÂGE
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

bins   = [0, 20, 30, 40, 50, 60, 70, 80, 100]
labels = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]

df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
grouped = df.groupby(['age_group', 'stroke'], observed=True).size().unstack(fill_value=0)
grouped.columns = ['Sans AVC', 'Avec AVC']

x     = np.arange(len(grouped))
width = 0.55

ax.bar(x, grouped['Sans AVC'], width, color=BLUE, alpha=0.75, label='Sans AVC')
ax.bar(x, grouped['Avec AVC'], width, color=RED,  alpha=0.90, label='Avec AVC',
       bottom=grouped['Sans AVC'])

for i, (sans, avec) in enumerate(zip(grouped['Sans AVC'], grouped['Avec AVC'])):
    total = sans + avec
    pct   = avec / total * 100 if total > 0 else 0
    if pct > 0:
        ax.text(i, sans + avec + 5, f"{pct:.0f}%",
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=RED)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Tranche d'âge", fontsize=12)
ax.set_ylabel("Nombre de patients", fontsize=12)
ax.set_title("🧠 Risque d'AVC selon l'âge", fontsize=15, fontweight='bold', color=DARK, pad=15)
ax.legend(frameon=False)
ax.text(0.5, -0.18, "💡 Le risque d'AVC augmente fortement après 50 ans",
        transform=ax.transAxes, ha='center', fontsize=11, color=BLUE, fontstyle='italic')

plt.tight_layout()
plt.savefig("graph1_avc_age.png", dpi=180, bbox_inches='tight')
plt.show()
print(" graph1_avc_age.png sauvegardé")

# ═══════════════════════════════════════════════════════════
#   GRAPHIQUE 2 — AVC vs HYPERTENSION & MALADIE CARDIAQUE
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("🫀 Facteurs cardiovasculaires et AVC",
             fontsize=15, fontweight='bold', color=DARK, y=1.02)

for ax, col, title in zip(
    axes,
    ['hypertension', 'heart_disease'],
    ['Hypertension', 'Maladie cardiaque']
):
    rates       = df.groupby(col)['stroke'].mean() * 100
    cat_labels  = ['Non', 'Oui']

    bars = ax.bar(cat_labels, rates.values, color=[BLUE, RED], alpha=0.85, width=0.5)

    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=DARK)

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel("Taux d'AVC (%)", fontsize=11)
    ax.set_ylim(0, rates.max() * 1.4)

    ratio = rates.values[1] / rates.values[0]
    ax.text(0.5, 0.88, f"×{ratio:.1f} de risque en plus",
            transform=ax.transAxes, ha='center', fontsize=10,
            color=RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', edgecolor=RED, alpha=0.8))

plt.tight_layout()
plt.savefig("graph2_avc_cardio.png", dpi=180, bbox_inches='tight')
plt.show()
print(" graph2_avc_cardio.png sauvegardé")

# ═══════════════════════════════════════════════════════════
#   GRAPHIQUE 3 — AVC vs TABAGISME
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

smoke_rate = (
    df.groupby('smoking_status')['stroke']
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

smoke_labels = {
    'smokes'         : '🚬 Fumeur actuel',
    'formerly smoked': '🚭 Ex-fumeur',
    'never smoked'   : '✅ Jamais fumé',
    'Unknown'        : '❓ Inconnu',
}
clean_labels = [smoke_labels.get(l, l) for l in smoke_rate.index]
palette      = [RED if "Fumeur" in l else (GRAY if "Inconnu" in l else BLUE)
                for l in clean_labels]

bars = ax.barh(clean_labels, smoke_rate.values, color=palette, alpha=0.85, height=0.5)

for bar, val in zip(bars, smoke_rate.values):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va='center', fontsize=12, fontweight='bold', color=DARK)

ax.set_xlabel("Taux d'AVC (%)", fontsize=12)
ax.set_title("🚬 Risque d'AVC selon le statut tabagique",
             fontsize=15, fontweight='bold', color=DARK, pad=15)
ax.set_xlim(0, smoke_rate.max() * 1.3)
ax.text(0.5, -0.15,
        "💡 Les fumeurs et ex-fumeurs ont un risque d'AVC significativement plus élevé",
        transform=ax.transAxes, ha='center', fontsize=10, color=BLUE, fontstyle='italic')

plt.tight_layout()
plt.savefig("graph3_avc_tabac.png", dpi=180, bbox_inches='tight')
plt.show()
print(" graph3_avc_tabac.png sauvegardé")

# ═══════════════════════════════════════════════════════════
#   MODÈLE ML — Logistic Regression
# ═══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  ENTRAÎNEMENT DU MODÈLE ML")
print("="*55)

df_ml = df.copy()
le    = LabelEncoder()
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
            'bmi', 'gender', 'ever_married', 'smoking_status']
X = df_ml[features]
y = df_ml['stroke']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy : {acc*100:.1f}%")
print("\n Rapport de classification :")
print(classification_report(y_test, y_pred, target_names=['Sans AVC', 'Avec AVC']))

# ── Graphique importance des features ──
coef_df = pd.DataFrame({
    'Feature'   : features,
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colors_imp = [RED if v > coef_df['Importance'].median() else BLUE
              for v in coef_df['Importance']]

ax.barh(coef_df['Feature'], coef_df['Importance'],
        color=colors_imp, alpha=0.85, height=0.55)
ax.set_title(f" Importance des variables — Logistic Regression\nAccuracy : {acc*100:.1f}%",
             fontsize=14, fontweight='bold', color=DARK, pad=12)
ax.set_xlabel("Importance (|coefficient|)", fontsize=11)

for val, name in zip(coef_df['Importance'], coef_df['Feature']):
    ax.text(val + 0.005, list(coef_df['Feature']).index(name),
            f"{val:.3f}", va='center', fontsize=9, color=DARK)

plt.tight_layout()
plt.savefig("graph4_ml_importance.png", dpi=180, bbox_inches='tight')
plt.show()
print(" graph4_ml_importance.png sauvegardé")

# ═══════════════════════════════════════════════════════════
#   INSIGHTS LINKEDIN
# ═══════════════════════════════════════════════════════════
hyp_ratio    = df[df['hypertension']==1]['stroke'].mean() / df[df['hypertension']==0]['stroke'].mean()
heart_ratio  = df[df['heart_disease']==1]['stroke'].mean() / df[df['heart_disease']==0]['stroke'].mean()

print("\n" + "="*55)
print("  INSIGHTS PRÊTS POUR LINKEDIN")
print("="*55)
print(f"""
 Analyse prédictive des AVC — {df.shape[0]} patients

 Ce que les données révèlent :
  • Le risque d'AVC explose après 50 ans
  • L'hypertension multiplie le risque par {hyp_ratio:.1f}x
  • La maladie cardiaque multiplie le risque par {heart_ratio:.1f}x
  • Les fumeurs ont un taux d'AVC plus élevé que les non-fumeurs

 Modèle ML :
  • Algorithme : Logistic Regression
  • Accuracy   : {acc*100:.1f}%
  • Features   : {len(features)} variables cliniques

 La data science peut aider à identifier les profils
   à risque AVANT l'accident cérébral.
""")
