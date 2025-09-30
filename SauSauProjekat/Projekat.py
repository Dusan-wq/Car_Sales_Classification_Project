import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore")

file = "BMW_Car_Sales_Classification.csv"
data = pd.read_csv(file)

data.dropna(inplace=True)  # uklanjamo prazne redove
data.drop_duplicates(inplace=True)  # uklanjamo duplikate

def detektujCudneVrednosti(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]
    print("Nakon uklanjanja outliera, broj redova:", len(df))
    return df

# eliminisemo besmislene i nelogicne podatke
data = data[(data['Mileage_KM'] >= 0) & (data['Price_USD'] > 0) & (data['Sales_Volume'] >= 0)]
data = detektujCudneVrednosti(data, ["Mileage_KM", "Price_USD", "Engine_Size_L", "Sales_Volume"])

le = LabelEncoder()
data['Color_encoded'] = le.fit_transform(data['Color'])
data['Model_encoded'] = le.fit_transform(data['Model'])
data['Region_encoded'] = le.fit_transform(data['Region'])
data['Fuel_encoded'] = le.fit_transform(data['Fuel_Type'])
data['Transmission_encoded'] = le.fit_transform(data['Transmission'])
data['Sales_Classification_encoded'] = le.fit_transform(data['Sales_Classification'])

dataframe = data[["Model_encoded", "Year", "Region_encoded", "Color_encoded", "Fuel_encoded",
                  "Transmission_encoded", "Engine_Size_L", "Mileage_KM", "Price_USD", "Sales_Volume",
                  "Sales_Classification_encoded"]]

matrix = dataframe.corr()
print("Korelaciona matrica:\n", matrix)

class_members = data['Sales_Classification'].value_counts()
print("Broj elemenata po klasama:\n", class_members)
numericki_podaci = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD']
korelacija = data[numericki_podaci + ['Sales_Classification_encoded']].corr()['Sales_Classification_encoded'].sort_values(ascending=False)
print("Korelacija numerickih atributa sa klasom:\n", korelacija)

price_bins = [0, 20000, 40000, 60000, 80000, 100000, 150000]
data['Price_bin'] = pd.cut(data['Price_USD'], bins=price_bins)
cm_price = pd.crosstab(data['Sales_Classification'], data['Price_bin'])
plt.figure(figsize=(12,6))
sns.heatmap(cm_price, cmap="Blues", annot=True, fmt="d")
plt.title("Sales Classification vs Price")
plt.ylabel("Sales Classification")
plt.xlabel("Price")
plt.show()

mileage_bins = [0, 20000, 40000, 60000, 80000, 100000, 150000, 200000]
data['Mileage_bin'] = pd.cut(data['Mileage_KM'], bins=mileage_bins)
cm_mileage = pd.crosstab(data['Sales_Classification'], data['Mileage_bin'])
plt.figure(figsize=(12,6))
sns.heatmap(cm_mileage, cmap="Greens", annot=True, fmt="d")
plt.title("Sales Classification vs Mileage")
plt.ylabel("Sales Classification")
plt.xlabel("Mileage")
plt.show()


X = data[["Model_encoded", "Year", "Region_encoded", "Color_encoded", "Fuel_encoded",
          "Transmission_encoded", "Engine_Size_L", "Mileage_KM", "Price_USD"]]
y = data["Sales_Classification_encoded"]

df_train = pd.concat([X, y], axis=1)
df_majority = df_train[df_train['Sales_Classification_encoded']==1]
df_minority = df_train[df_train['Sales_Classification_encoded']==0]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)
df_train_balanced = pd.concat([df_majority, df_minority_upsampled])
X_balanced = df_train_balanced.drop('Sales_Classification_encoded', axis=1)
y_balanced = df_train_balanced['Sales_Classification_encoded']

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train[numericki_podaci] = scaler.fit_transform(X_train[numericki_podaci])
X_test[numericki_podaci] = scaler.transform(X_test[numericki_podaci])

def tune_model(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"\n--- {model_name} ---")
    print("Najbolji hiperparametri:", grid.best_params_)
    print("Accuracy na test skupu:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    if hasattr(best_model, 'feature_importances_'):
        for col, imp in zip(X_train.columns, best_model.feature_importances_):
            print(f"{col}: {imp:.3f}")
    return best_model

param_grid_lr = {"C": [0.01,0.1,1], "penalty": ["l2"]}
najbolji_lr = tune_model(LogisticRegression(class_weight="balanced", max_iter=500),
                         param_grid_lr, X_train, y_train, X_test, y_test, "Logistic Regression")

param_grid_dt = {"max_depth":[3,6,12], "criterion":["gini","entropy"]}
najbolji_dt = tune_model(DecisionTreeClassifier(class_weight="balanced"),
                         param_grid_dt, X_train, y_train, X_test, y_test, "Decision Tree")

param_grid_rf = {"n_estimators":[50,100,200], "max_depth":[None,6,12], "criterion":["gini","entropy"]}
najbolji_rf = tune_model(RandomForestClassifier(random_state=42, class_weight="balanced"),
                         param_grid_rf, X_train, y_train, X_test, y_test, "Random Forest")

param_grid_xgb = {"n_estimators":[50,100,200], "max_depth":[3,6,10], "learning_rate":[0.05,0.1]}
najbolji_xgb = tune_model(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                         param_grid_xgb, X_train, y_train, X_test, y_test, "XGBoost")

models = [("Random Forest", najbolji_rf),
          ("Logistic Regression", najbolji_lr),
          ("Decision Tree", najbolji_dt),
          ("XGBoost", najbolji_xgb)]

for name, model in models:
    y_pred = model.predict(X_test)
    print(f"{name} | Accuracy: {accuracy_score(y_test, y_pred):.3f} | F1-score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

importances = najbolji_rf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values(by="Importance", ascending=False)
najbitniji = feat_imp_df["Feature"].head(5).tolist()
X_bitni = X[najbitniji]

top2_features = najbitniji[:2]

plt.figure(figsize=(10,7))

for i, class_name in enumerate(class_members.index):
    plt.scatter(
        X_bitni.loc[y == i, top2_features[0]],
        X_bitni.loc[y == i, top2_features[1]],
        label=class_name,
        alpha=0.6,
        s=3
    )

plt.xscale('log')
plt.yscale('log')
plt.xlabel(top2_features[0])
plt.ylabel(top2_features[1])
plt.title(f"BMW Dataset - {top2_features[0]} vs {top2_features[1]} (log scale)")
plt.legend()
plt.grid()
plt.show()

# Top 5 atributa sa uravnoteženim skupom
X_top5 = X_balanced[najbitniji]
y_top5 = y_balanced

X_train_top5, X_test_top5, y_train_top5, y_test_top5 = train_test_split(
    X_top5, y_top5, test_size=0.2, random_state=42
)

# Skaliranje numeričkih kolona
num_cols_top5 = [col for col in numericki_podaci if col in X_top5.columns]
scaler_top5 = StandardScaler()
X_train_top5[num_cols_top5] = scaler_top5.fit_transform(X_train_top5[num_cols_top5])
X_test_top5[num_cols_top5] = scaler_top5.transform(X_test_top5[num_cols_top5])

# Modeli
xgb_params = najbolji_xgb.get_params()
xgb_params.update({'use_label_encoder': False, 'eval_metric': 'logloss'})
models_top5 = {
    "Random Forest": RandomForestClassifier(**najbolji_rf.get_params()),
    "Logistic Regression": LogisticRegression(**najbolji_lr.get_params()),
    "Decision Tree": DecisionTreeClassifier(**najbolji_dt.get_params()),
    "XGBoost": XGBClassifier(**xgb_params)
}

# Evaluacija
for name, model in models_top5.items():
    model.fit(X_train_top5, y_train_top5)
    y_pred_top5 = model.predict(X_test_top5)
    acc = accuracy_score(y_test_top5, y_pred_top5)
    f1 = f1_score(y_test_top5, y_pred_top5, average='weighted')
    print(f"\n{name} - Top 5 atributa")
    print(f"Accuracy: {acc:.3f} | F1-score: {f1:.3f}")
    print("Classification Report:\n", classification_report(y_test_top5, y_pred_top5, zero_division=0))
