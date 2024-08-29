import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Verificar las primeras filas de los datos para entender su estructura
print("Primeras filas del DataFrame de entrenamiento:")
print(train.head())
print("Primeras filas del DataFrame de prueba:")
print(test.head())

# Verificar la información y los tipos de datos
print("Información del DataFrame de entrenamiento:")
print(train.info())
print("Información del DataFrame de prueba:")
print(test.info())

# Verificar la estadística descriptiva de los datos
print("Estadística descriptiva del DataFrame de entrenamiento:")
print(train.describe())
print("Estadística descriptiva del DataFrame de prueba:")
print(test.describe())

# Tratar valores faltantes si los hay
train = train.dropna()
test = test.dropna()

# Verificar columnas después de eliminar valores faltantes
print("Columnas del DataFrame de entrenamiento después de eliminar valores faltantes:")
print(train.columns)
print("Columnas del DataFrame de prueba después de eliminar valores faltantes:")
print(test.columns)

# Convertir variables categóricas en numéricas usando un LabelEncoder separado para cada columna
label_encoders = {}
for column in ['Distributor', 'Product', 'Destination', 'Gender']:
    if column in train.columns:
        label_encoder = LabelEncoder()
        label_encoder.fit(train[column])
        label_encoders[column] = label_encoder
        train[column] = label_encoder.transform(train[column])
        test[column] = test[column].apply(lambda x: x if x in label_encoder.classes_ else 'UNKNOWN')
        test[column] = test[column].map(lambda x: label_encoder.transform([x])[0] if x in label_encoder.classes_ else -1)
    else:
        print(f"Columna {column} no encontrada en el DataFrame de entrenamiento")

X_train = train.drop(['ID', 'Target'], axis=1, errors='ignore')
y_train = train['Target']

X_test = test.drop(['ID'], axis=1, errors='ignore')

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train_split, y_train_split)

y_val_pred = model.predict(X_val_split)

print(f'F1 Score: {f1_score(y_val_split, y_val_pred, average="weighted")}')

importances = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Importancia de las características:")
print(importance_df)

# Visualización de la importancia de las características
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Importancia de las Características')
plt.show()

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'ID': test['ID'],
    'Target': test_predictions
})

# Guardar el archivo de envío
submission.to_csv('submission.csv', index=False)



