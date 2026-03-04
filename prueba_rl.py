#El codigo evalua el modelo Regresion Logistica utilizando 5 metricas de clasificacion y validacion cruzada

#Bibliotecas a utilizar
import pandas as pd #pandas, sirve para leer, limpiar, manipular y explorar datasets
import numpy as np #numpy, es necesaria para scikit-learn, realiza los calculos del modelo
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_validate #split 70/30 y validacion cruzada
from sklearn.preprocessing import StandardScaler,OneHotEncoder #preprocesamiento
from sklearn.compose import ColumnTransformer #transformaciones por tipo de columna
from sklearn.pipeline import Pipeline #pipeline completo
from sklearn.impute import SimpleImputer #manejo de faltantes
from sklearn.linear_model import LogisticRegression #modelo regresion logistica
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,roc_curve #metricas y graficas
import matplotlib.pyplot as plt #graficas

#cargamos el dataset
df=pd.read_csv("german_credit_data.csv")

#limpieza ligera de nombres
df.columns=df.columns.str.strip().str.lower().str.replace(" ","_")

#variable objetivo
target_col="risk"
df[target_col]=df[target_col].astype(str).str.lower().map({"good":0,"bad":1})

#si quedaron nulos por etiquetas raras, se eliminan
df=df.dropna(subset=[target_col])

X=df.drop(columns=[target_col])
y=df[target_col].astype(int)

#division 70/30 con estratificacion
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.30,stratify=y,random_state=42
)

#se detectan columnas numericas y categoricas
num_cols=X.select_dtypes(include=np.number).columns.tolist()
cat_cols=X.select_dtypes(exclude=np.number).columns.tolist()

#pipeline numerico
num_proc=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

#pipeline categorico
cat_proc=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

#transformador general
preprocessor=ColumnTransformer([
    ("num",num_proc,num_cols),
    ("cat",cat_proc,cat_cols)
])

#modelo regresion logistica (se aumenta max_iter para asegurar convergencia)
log_model=Pipeline([
    ("preprocessor",preprocessor),
    ("logreg",LogisticRegression(max_iter=1000,random_state=42))
])

#validacion cruzada (solo en train)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scoring={"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}

cv_results=cross_validate(log_model,X_train,y_train,cv=cv,scoring=scoring,n_jobs=-1)

print("=====Validacion cruzada (5-fold,estratificada) sobre TRAIN=====")
for m in scoring:
    vals=cv_results["test_"+m]
    print(f"{m}:mean={vals.mean():.3f},std={vals.std():.3f}")

#entrenamiento final en train y evaluacion en test
log_model.fit(X_train,y_train)
y_pred=log_model.predict(X_test)
y_proba=log_model.predict_proba(X_test)[:,1]

acc=accuracy_score(y_test,y_pred)
pre=precision_score(y_test,y_pred,pos_label=1,zero_division=0)
rec=recall_score(y_test,y_pred,pos_label=1,zero_division=0)
f1=f1_score(y_test,y_pred,pos_label=1,zero_division=0)
auc=roc_auc_score(y_test,y_proba)

print("\n=====Resultados en conjunto de prueba (30%)=====")
print(f"Accuracy:{acc:.3f}")
print(f"Precision:{pre:.3f}")
print(f"Recall:{rec:.3f}")
print(f"F1-score:{f1:.3f}")
print(f"ROC-AUC:{auc:.3f}")

cm=confusion_matrix(y_test,y_pred)
print("\nMatriz de confusion TEST [ [TN, FP], [FN, TP] ]:")
print(cm)

#matriz de confusion normalizada (test)
ConfusionMatrixDisplay.from_predictions(
    y_test,y_pred,
    display_labels=["Bueno (0)","Riesgoso (1)"],
    normalize="true",cmap="Blues",values_format=".2f"
)
plt.title("Matriz de confusion normalizada - Regresion Logistica (test 30%)")
plt.tight_layout()
plt.savefig("logistic_confusion_matrix_normalized.png",dpi=200)
plt.show()

#curva roc (test)
fpr,tpr,thr=roc_curve(y_test,y_proba)
plt.figure()
plt.plot(fpr,tpr,label=f"Logistica (AUC={auc:.3f})")
plt.plot([0,1],[0,1],linestyle="--",label="Azar")
plt.xlabel("FPR (False Positive Rate)")
plt.ylabel("TPR (Recall)")
plt.title("Curva ROC - Regresion Logistica (test 30%)")
plt.legend()
plt.tight_layout()
plt.savefig("logistic_roc_curve.png",dpi=200)
plt.show()