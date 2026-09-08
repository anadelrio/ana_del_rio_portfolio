import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import randint, uniform, linregress
import os
import matplotlib.pyplot as plt
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

# 1. Loading and merge of the data -------------------------------------------------------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "datasets" / "inputs"
df_main = pd.read_csv(DATA_DIR / "liga_hypermotion_2020_2025_limpio2.csv", parse_dates=["date"])
team_loc = pd.read_csv(DATA_DIR / "team_locations.csv")
df_val = pd.read_csv(DATA_DIR / "team_market_value_season.csv")
df_val_mean = df_val.groupby('team')['market_millions'].mean().reset_index()

df = (
    df_main
    .merge(team_loc.rename(columns={'team':'home_team','lat':'home_latitude','lon':'home_longitude'}), on='home_team', how='left')
    .merge(team_loc.rename(columns={'team':'away_team','lat':'away_latitude','lon':'away_longitude'}), on='away_team', how='left')
    .merge(df_val_mean.rename(columns={'team':'home_team','market_millions':'home_market_millions'}), on='home_team', how='left')
    .merge(df_val_mean.rename(columns={'team':'away_team','market_millions':'away_market_millions'}), on='away_team', how='left')
)

# 2. Feature engineering --------------------------------------------------------------------------------------------------
W = 5
hp_map = {'H':3,'D':1,'A':0}
for side, win_label in [('home','H'),('away','A')]:
    pts_col = f"{side}_form_points_last{W}"
    wr_col  = f"{side}_form_winrate_last{W}"
    cc_col  = f"{side}_conceded_last{W}"
    df[pts_col] = df.groupby(f"{side}_team")['result']\
        .transform(lambda s: s.shift().map(hp_map).rolling(W,min_periods=1).mean())
    df[wr_col] = df.groupby(f"{side}_team")['result']\
        .transform(lambda s: s.shift().eq(win_label).rolling(W,min_periods=1).mean())
    goals   = 'away_goals' if side=='home' else 'home_goals'
    df[cc_col] = df.groupby(f"{side}_team")[goals]\
        .transform(lambda s: s.shift().rolling(W,min_periods=1).mean())
    trend_col = f"{side}_trend_points_last{W}"
    df[trend_col] = df.groupby(f"{side}_team")[pts_col]\
        .transform(lambda x: x.rolling(W, min_periods=2)\
                             .apply(lambda y: linregress(np.arange(len(y)),y).slope, raw=True))
# Interactions
df['int_points_mul']    = df['home_form_points_last5'] * df['away_form_points_last5']
df['int_conceded_diff'] = df['home_conceded_last5'] - df['away_conceded_last5']

# 3. Preparation of the data -----------------------------------------------------------------------------------------------------------------------------------------
df['result_enc'] = LabelEncoder().fit_transform(df['result'])
exclude = ['result','result_enc','date','home_team','away_team','home_goals','away_goals','bet_odds_home','bet_odds_draw','bet_odds_away','home_is_favorite']
features = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
X = df[features]
y = df['result_enc']

print("\nUnique classes across the entire dataset:")
print(df['result'].value_counts())
print("Numeric labels:", df['result_enc'].value_counts())

# 4. Split Train+Val vs Test -----------------------------------------------------------------------------------------
X_trv, X_test, y_trv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print("\nClasses in Train+Val:", pd.Series(y_trv).value_counts())
print("Classes in Test:", pd.Series(y_test).value_counts())

# 5. Preprocessing ------------------------------------------------------------------------------------------------------------------------------------
pipeline = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc',  StandardScaler())
])
X_trv_s  = pipeline.fit_transform(X_trv)
X_test_s = pipeline.transform(X_test)

# 6. Definition and adjustment of the baseline models ------------------------------------------------------------------------------------------------------------------------------
# Random Forest 
rf_dist = {'n_estimators': randint(300,600), 'max_depth': randint(6,30), 'min_samples_split': randint(2,10), 'min_samples_leaf': randint(1,8), 'max_features': uniform(0.3,0.7)}
rf_cv   = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    rf_dist,
    n_iter=15,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # 5 folds
    scoring='f1_macro',
    n_jobs=-1,
    random_state=42
)
# Other models
models = {
    'Random Forest': rf_cv,
    'Gaussian NB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'SVM RBF': SVC(kernel='rbf',probability=True,random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=2000,multi_class='multinomial',class_weight='balanced',random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(128,64),max_iter=300,random_state=42),
    'LightGBM': LGBMClassifier(objective='multiclass',n_estimators=200,learning_rate=0.05,num_leaves=63,subsample=0.8,colsample_bytree=0.8,class_weight='balanced',random_state=42,verbose=-1)
}

trained = {}
for name, mdl in models.items():
    try:
        if name == 'Random Forest':
            rf_cv.fit(X_trv_s, y_trv)
            trained[name] = rf_cv.best_estimator_
        else:
            mdl.fit(X_trv_s, y_trv)
            trained[name] = mdl
    except Exception as e:
        print(f"Skipping {name} due to training error: {e}")
models = trained

# 7. Selection of the best model -----------------------------------------------------------------------------------------------------------------------------
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
f1_scores = {}
for name, mdl in models.items():
    avg = cross_validate(Pipeline([('prep',pipeline),('clf',mdl)]), X_trv, y_trv, cv=kf, scoring=make_scorer(f1_score,average='macro'), n_jobs=-1)['test_score'].mean()
    f1_scores[name] = avg
best_model_name = max(f1_scores, key=f1_scores.get)
best_model = models[best_model_name]
print(f"Best model: {best_model_name}")

# Function to generate predictive sets
def predict_set(probs, q_hat):
    """
    Given a probability matrix of shape (n_samples, n_classes)
    and a threshold q_hat, returns a boolean mask indicating
    which classes are part of the prediction set.
    """
    return (1 - probs) <= q_hat

trained = {}
for name, mdl in models.items():
    try:
        if name == 'Random Forest':
            rf_cv.fit(X_trv_s, y_trv)
            trained[name] = rf_cv.best_estimator_
        else:
            mdl.fit(X_trv_s, y_trv)
            trained[name] = mdl
    except Exception as e:
        print(f"Skipping {name} due to training error: {e}")
models = trained

# Auxiliar function to compute conformal threshold
def calibrate_q(probs, y_true, classes, alpha=0.1):
    """
    Calculates the threshold q̂ for Conformal Prediction,
    given a probability array (n_samples, n_classes),
    the true labels and the list of classes.
    """
    idx = {c: i for i, c in enumerate(classes)}
    scores = 1 - probs[np.arange(len(y_true)), [idx[y] for y in y_true]]
    return np.quantile(scores, 1 - alpha)
unique_classes = np.unique(y_trv)
if len(unique_classes) > 1:

# 8. Calibration with 5-fold CV over all the train ------------------------------------------------------------------------------------------------------------------
    calibrator = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
    calibrator.fit(X_trv_s, y_trv)
    print("Successful isotonic calibration with CV=5.")
    
    # Compute q̂ threshold for conformal prediction
    alpha = 0.10
    q_hat = calibrate_q(
        calibrator.predict_proba(X_trv_s),
        y_trv,
        calibrator.classes_,
        alpha
    )
    print(f"q̂ (α={alpha}): {q_hat:.4f}")

else:
    # Fallback to model without calibration
    print(
        "Attention! There is only one class after sampling."
        f"(class {unique_classes[0]}). Calibration is skipped."
    )
    calibrator = best_model
    # In this case, q̂ = 0 by definition (all probabilities = 1)
    q_hat = 0.0
    print(f"q̂ (fallback): {q_hat:.4f}")

# 9. Test evaluation ------------------------------------------------------------------------------------------------------------------------
probs_test = calibrator.predict_proba(X_test_s)
y_pred = calibrator.predict(X_test_s)
print("=== Test Results ===")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test,y_pred):.4f}")
print(f"Recall_macro: {recall_score(y_test,y_pred,average='macro'):.4f}")
print(f"F1_macro: {f1_score(y_test,y_pred,average='macro'):.4f}")
print(f"AUC-ROC_macro: {roc_auc_score(y_test,probs_test,multi_class='ovr'):.4f}")
# Conformal sets
mask = predict_set(probs_test, q_hat)
print(f"Empirical coverage: {mask[np.arange(len(y_test)),y_test].mean():.4f}")
print(f"Average set size: {mask.sum(axis=1).mean():.4f}")

# 10. Feature importance  -----------------------------------------------------------------------------------------------------------------------------------------------------
# Extract coefficients/importances and names (works for linear AND tree-based models)
feature_names = features
base_estimator = calibrator.calibrated_classifiers_[0].estimator

if hasattr(base_estimator, "coef_"):
    # Linear models (e.g. Logistic Regression): per-class coefficients
    coefs_list = [clf.estimator.coef_ for clf in calibrator.calibrated_classifiers_]
    avg_coefs = np.mean(coefs_list, axis=0)  # Shape: (n_classes, n_features)

    for i, class_name in enumerate(calibrator.classes_):
        sorted_idx = np.argsort(np.abs(avg_coefs[i]))[::-1]
        top_features = [feature_names[j] for j in sorted_idx[:10]]
        top_values = avg_coefs[i, sorted_idx[:10]]

        plt.figure(figsize=(8, 5))
        plt.barh(top_features[::-1], top_values[::-1])
        plt.title(f"Top 10 features for class: {class_name}")
        plt.xlabel("Average coefficient (impact in log-odds)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

elif hasattr(base_estimator, "feature_importances_"):
    # Tree-based models (e.g. Random Forest, LightGBM): single global importance
    importances_list = [clf.estimator.feature_importances_ for clf in calibrator.calibrated_classifiers_]
    avg_importances = np.mean(importances_list, axis=0)
    sorted_idx = np.argsort(avg_importances)[::-1]
    top_features = [feature_names[j] for j in sorted_idx[:10]]
    top_values = avg_importances[sorted_idx[:10]]

    plt.figure(figsize=(8, 5))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.title(f"Top 10 features — {best_model_name}")
    plt.xlabel("Average feature importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print(f"Feature importance visualization not available for model type: {type(base_estimator).__name__}")


# 11. Interactive prediction -----------------------------------------------------------------------------------------------------------------------------------------------------------------
print("=== Interactive Prediction ===")
home = input("Home Team: ")
away = input("Away Team: ")
date_str = input("Match Date (YYYY-MM-DD): ")
date = datetime.strptime(date_str, "%Y-%m-%d")
new = pd.DataFrame([{'home_team':home,'away_team':away,'date':date}])
# Merge y feature engineering idénticos a pasos 1-3
new['home_team'],new['away_team']=new['home_team'].astype(str),new['away_team'].astype(str)
new = new.merge(
    team_loc.rename(columns={'team':'home_team','lat':'home_latitude','lon':'home_longitude'}), on='home_team', how='left'
).merge(
    team_loc.rename(columns={'team':'away_team','lat':'away_latitude','lon':'away_longitude'}), on='away_team', how='left'
).merge(
    df_val.rename(columns={'team':'home_team','market_millions':'home_market_millions'}), on='home_team', how='left'
).merge(
    df_val.rename(columns={'team':'away_team','market_millions':'away_market_millions'}), on='away_team', how='left'
)
for side, wl in [('home','H'),('away','A')]:
    hist = df[(df[f"{side}_team"]==new[f"{side}_team"].iloc[0]) & (df['date']<date)]
    if not hist.empty:
        new[f"{side}_form_points_last{W}"]   = hist[f"{side}_form_points_last{W}"].iloc[-1]
        new[f"{side}_form_winrate_last{W}"] = hist[f"{side}_form_winrate_last{W}"].iloc[-1]
        new[f"{side}_conceded_last{W}"]     = hist[f"{side}_conceded_last{W}"].iloc[-1]
        new[f"{side}_trend_points_last{W}"] = hist[f"{side}_trend_points_last{W}"].iloc[-1]
    else:
        for col in [
            f"{side}_form_points_last{W}", f"{side}_form_winrate_last{W}",
            f"{side}_conceded_last{W}", f"{side}_trend_points_last{W}"
        ]:
            new[col] = 0
new['int_points_mul']    = new['home_form_points_last5'] * new['away_form_points_last5']
new['int_conceded_diff'] = new['home_conceded_last5'] - new['away_conceded_last5']
X_new = pd.DataFrame(0, index=[0], columns=features)
common = [c for c in features if c in new.columns]
X_new.loc[0, common] = new.loc[0, common]
X_new = X_new.fillna(0)
X_new_s = pipeline.transform(X_new)
probs_new = calibrator.predict_proba(X_new_s)[0]
print("\nProbabilities:")
for cls, p in zip(calibrator.classes_, probs_new): print(f"- {cls}: {p:.3f}")
pred_set = [cls for cls, m in zip(calibrator.classes_, predict_set(probs_new.reshape(1,-1),q_hat)[0]) if m]
print("Predictive Set (Conformal):", pred_set)