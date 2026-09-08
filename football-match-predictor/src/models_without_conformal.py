import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from lightgbm import LGBMClassifier, early_stopping
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, make_scorer, classification_report, confusion_matrix
from scipy.stats import randint, uniform, linregress
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import recall_score, roc_auc_score
import os

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings

# Silence every warning
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=FutureWarning)

# 1. Load and data merge ---------------------------------------------------------------------------------------------------------------------------------------
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

# 2. Feature engineering ----------------------------------------------------------------------------------------------------------------------
W = 5
hp_map = {"H":3, "D":1, "A":0}
# Compute the metrics that are known in the rolling window
for side, win_label in [('home','H'), ('away','A')]:
    pts_col = f"{side}_form_points_last{W}"
    wr_col  = f"{side}_form_winrate_last{W}"
    con_col = f"{side}_conceded_last{W}"
    # Rolling mean of points (map HR labels) and winrate
    df[pts_col] = df.groupby(f"{side}_team")['result']\
        .transform(lambda s: s.shift().map(hp_map).rolling(W, min_periods=1).mean())
    df[wr_col]  = df.groupby(f"{side}_team")['result']\
        .transform(lambda s: s.shift().eq(win_label).rolling(W, min_periods=1).mean())
    # Rolling mean conceeded goals
    conceded = 'away_goals' if side=='home' else 'home_goals'
    df[con_col] = df.groupby(f"{side}_team")[conceded]\
        .transform(lambda s: s.shift().rolling(W, min_periods=1).mean())
    # Tendencia (pendiente) de la forma en puntos en ventana W
    trend_col = f"{side}_trend_points_last{W}"
    df[trend_col] = df.groupby(f"{side}_team")[pts_col]\
        .transform(lambda x: x.rolling(W, min_periods=2)\
                              .apply(lambda y: linregress(np.arange(len(y)), y).slope, raw=True))

# Interaction variables between home and away
df['int_points_mul'] = df['home_form_points_last5'] * df['away_form_points_last5']
df['int_conceded_diff'] = df['home_conceded_last5'] - df['away_conceded_last5']


# 3. Preparation of labels and final features ----------------------------------------------------------------------------------
df['result_enc'] = LabelEncoder().fit_transform(df['result'])
exclude = [
    'result','result_enc','date','home_team','away_team',
    'home_goals','away_goals',
    'bet_odds_home','bet_odds_draw','bet_odds_away','home_is_favorite'
]
num_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
X = df[num_feats]
y = df['result_enc']

# 4. Split: train+val vs test for final evaluation --------------------------------------------------------------------------------------------------------
X_trv, X_test, y_trv, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 5. Validation: Stratified K-Fold CV ------------------------------------------------------------------------------------------------------------------------------
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

preproc = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 6. Definition of baseline models and ensemble --------------------------------------------------------------------------------------------------------
# Random Forest 
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_params = {
    'n_estimators': randint(300,600),
    'max_depth': randint(6,30),
    'min_samples_split': randint(2,10),
    'min_samples_leaf': randint(1,8),
    'max_features': uniform(0.3,0.7)
}
rf_search = RandomizedSearchCV(
    rf, rf_params, n_iter=15,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring='f1_macro', n_jobs=-1, random_state=42
)
rf_search.fit(preproc.fit_transform(X_trv), y_trv)
rf_best = rf_search.best_estimator_

# Aliditional base model
gnb_model = GaussianNB()  
lda_model = LinearDiscriminantAnalysis()  
qda_model = QuadraticDiscriminantAnalysis()  
svc_model = SVC(kernel='rbf', probability=True, random_state=42) 
log_model = LogisticRegression(max_iter=2000, multi_class='multinomial', class_weight='balanced', random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=42)
# LightGBM with early stopping
gb_model = LGBMClassifier(
    objective='multiclass', n_estimators=200, learning_rate=0.05,
    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced', random_state=42, verbose=-1
)
# Split manual for early stopping
tmp_X_tr, tmp_X_val, tmp_y_tr, tmp_y_val = train_test_split(
    X_trv, y_trv, test_size=0.1, random_state=42, stratify=y_trv
)
X_tr_s = preproc.fit_transform(tmp_X_tr)
X_val_s = preproc.transform(tmp_X_val)
gb_model.fit(
    X_tr_s, tmp_y_tr,
    eval_set=[(X_val_s, tmp_y_val)], eval_metric='multi_logloss',
    callbacks=[early_stopping(stopping_rounds=20)]
)

models = {
    'Random Forest': rf_best,
    'Gaussian NB': gnb_model,
    'LDA': lda_model,
    'QDA': qda_model,
    'SVM RBF': svc_model,
    'Logistic Regression': log_model,
    'MLP Neural Net': mlp_model,
    'LightGBM': gb_model
}

# 7. Identification of the best model after CV to use in the decomposition of the error -----------------------------------------------------------------------------------
cv_macro = {}
for name, mdl in models.items():
    pipe = Pipeline([('prep', preproc), ('clf', mdl)])
    scores = cross_validate(
        pipe, X_trv, y_trv,
        cv=kf, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1
    )
    cv_macro[name] = scores['test_score'].mean()
best_model_name = max(cv_macro, key=cv_macro.get)
best_model = models[best_model_name]
print(f"Best model for bias-variance decomposition: {best_model_name}")

# 8. Decomposition of the error (Bias-Variance) for the best model -------------------------------------------------------------------------------------------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
pred_dict = {i: [] for i in range(len(X_trv))}
for train_idx, test_idx in rskf.split(X_trv, y_trv):
    model_fold = Pipeline([('prep', preproc), ('clf', best_model)])
    model_fold.fit(X_trv.iloc[train_idx], y_trv.iloc[train_idx])
    preds = model_fold.predict(X_trv.iloc[test_idx])
    for idx, pred in zip(test_idx, preds):
        pred_dict[idx].append(pred)
# Compute bias y variance
y_true = y_trv.reset_index(drop=True)
bias_sq_list = []
var_list = []
for i, preds in pred_dict.items():
    if not preds:
        continue
    mean_pred = np.mean(preds)
    bias_sq_list.append((mean_pred - y_true[i])**2)
    var_list.append(np.mean((preds - mean_pred)**2))
# Report
print(f"Mean of Bias^2 (training): {np.mean(bias_sq_list):.4f}")
print(f"Mean of  variance (training): {np.mean(var_list):.4f}")

# 9. Final evaluation in test for all the models and resume of the best -------------------------------------------------------------------
for name, mdl in models.items():
    pipe = Pipeline([('prep', preproc), ('clf', mdl)])
    pipe.fit(X_trv, y_trv)
    y_pred = pipe.predict(X_test)
    try:
        y_prob = pipe.predict_proba(X_test)
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except Exception:
        auc_score = None
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, average='macro')
    f1m = f1_score(y_test, y_pred, average='macro')
    print(f"--- {name} in Test ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"Recall_macro: {rec:.3f}")
    print(f"F1_macro: {f1m:.3f}")
    if auc_score is not None:
        print(f"AUC-ROC_macro: {auc_score:.3f}")
    print()

best_model_name = max(models, key=lambda n: cross_validate(
    Pipeline([('prep', preproc), ('clf', models[n])]),
    X_trv, y_trv, cv=kf, scoring=make_scorer(f1_score, average='macro'), n_jobs=-1
)['test_score'].mean())
best_model = models[best_model_name]
print(f"Best model after CV: {best_model_name}")
print("Final evaluation on test set:")
best_pipe = Pipeline([('prep', preproc), ('clf', best_model)])
best_pipe.fit(X_trv, y_trv)
y_pred_best = best_pipe.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.3f}")
print(f"Recall_macro: {recall_score(y_test, y_pred_best, average='macro'):.3f}")
print(f"F1_macro: {f1_score(y_test, y_pred_best, average='macro'):.3f}")
y_prob_best = best_pipe.predict_proba(X_test)
print(f"AUC-ROC_macro: {roc_auc_score(y_test, y_prob_best, multi_class='ovr'):.4f}")
