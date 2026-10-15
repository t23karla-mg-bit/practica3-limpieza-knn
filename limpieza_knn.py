import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

print("=== INICIANDO PIPELINE DE LIMPIEZA Y PREPROCESAMIENTO ===")

data = {
    "edad": [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40],
    "ingreso_mensual": [15000, 22000, 18000, np.nan, 12000, 950000, 25000, 21000, 19000, np.nan],
    "scoring_credito": [650, 700, 680, 710, 590, 800, 670, 640, 620, 730]
}
df_dirty = pd.DataFrame(data)

print("\n--- DATASET ORIGINAL (CON RUIDO Y NULOS) ---")
print(df_dirty)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_dirty["ingreso_mensual"], color="salmon")
plt.title("Ingreso Mensual (Con Outlier)")

plt.subplot(1, 2, 2)
sns.boxplot(y=df_dirty["edad"], color="skyblue")
plt.title("Edad (Con Erroneo)")
plt.tight_layout()
plt.savefig("boxplot_antes.png")
plt.close()

Q1 = df_dirty["ingreso_mensual"].quantile(0.25)
Q3 = df_dirty["ingreso_mensual"].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 1.5 * IQR

print(f"\n[IQR] Limite Superior para Ingreso Mensual: ${limite_superior:,.2f}")

df_dirty.loc[df_dirty["ingreso_mensual"] > limite_superior, "ingreso_mensual"] = np.nan
df_dirty.loc[df_dirty["edad"] > 100, "edad"] = np.nan

print("\n--- DATASET TRAS REEMPLAZO DE OUTLIERS POR NaN ---")
print(df_dirty)

imputer_k2 = KNNImputer(n_neighbors=2)
df_clean_k2 = pd.DataFrame(imputer_k2.fit_transform(df_dirty), columns=df_dirty.columns)

print("\n=== DATASET LIMPIO E IMPUTADO (KNN k=2) ===")
print(df_clean_k2)

imputer_k5 = KNNImputer(n_neighbors=5)
df_clean_k5 = pd.DataFrame(imputer_k5.fit_transform(df_dirty), columns=df_dirty.columns)

df_media = df_dirty.fillna(df_dirty.mean())
df_mediana = df_dirty.fillna(df_dirty.median())

print("\n=== COMPARATIVA DE VALORES IMPUTADOS PARA ingreso_mensual ===")
comparativa = pd.DataFrame({
    "Original_NaN": df_dirty["ingreso_mensual"],
    "KNN_k2": df_clean_k2["ingreso_mensual"],
    "KNN_k5": df_clean_k5["ingreso_mensual"],
    "Imput_Media": df_media["ingreso_mensual"],
    "Imput_Mediana": df_mediana["ingreso_mensual"]
})
print(comparativa)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_clean_k2["ingreso_mensual"], color="lightgreen")
plt.title("Ingreso Mensual (Tratado k=2)")

plt.subplot(1, 2, 2)
sns.boxplot(y=df_clean_k2["edad"], color="lightgreen")
plt.title("Edad (Tratada k=2)")
plt.tight_layout()
plt.savefig("boxplot_despues.png")
plt.close()

print("\nPipeline completado. Graficos creados exitosamente.")
