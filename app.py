from flask import Flask, render_template_string, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings
import io
import base64

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Génération des données de ventes
def generate_sales_data(date_range):
    data = []
    for i, date in enumerate(date_range):
        base_sales = 1000 + (i * 2)
        day_of_week = date.weekday()
        weekly_multiplier = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 1.3][day_of_week]
        month_multiplier = [0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 0.9, 0.8, 1.1, 1.3, 1.5, 1.8][date.month - 1]
        noise = np.random.normal(0, 100)
        sales = base_sales * weekly_multiplier * month_multiplier + noise
        data.append({
            'date': date,
            'sales': max(0, sales),
            'day_of_week': day_of_week,
            'month': date.month,
            'is_weekend': day_of_week >= 5
        })
    return pd.DataFrame(data)

def inject_anomalies(df):
    df_copy = df.copy()
    black_friday = pd.to_datetime('2023-11-24')
    df_copy.loc[df_copy['date'] == black_friday, 'sales'] *= 3
    df_copy.loc[df_copy['date'] == black_friday, 'anomaly_type'] = 'black_friday'
    stock_out_dates = ['2023-03-15', '2023-08-22', '2024-01-10']
    for date_str in stock_out_dates:
        date = pd.to_datetime(date_str)
        df_copy.loc[df_copy['date'] == date, 'sales'] *= 0.2
        df_copy.loc[df_copy['date'] == date, 'anomaly_type'] = 'stock_out'
    promo_dates = ['2023-07-14', '2023-12-26', '2024-02-14']
    for date_str in promo_dates:
        date = pd.to_datetime(date_str)
        df_copy.loc[df_copy['date'] == date, 'sales'] *= 2.5
        df_copy.loc[df_copy['date'] == date, 'anomaly_type'] = 'promotion'
    tech_issue_dates = ['2023-05-10', '2023-09-18']
    for date_str in tech_issue_dates:
        date = pd.to_datetime(date_str)
        df_copy.loc[df_copy['date'] == date, 'sales'] *= 0.1
        df_copy.loc[df_copy['date'] == date, 'anomaly_type'] = 'tech_issue'
    df_copy['anomaly_type'].fillna('normal', inplace=True)
    return df_copy

def create_features(df):
    df_features = df.copy()
    df_features['day_of_year'] = df_features['date'].dt.dayofyear
    df_features['week_of_year'] = df_features['date'].dt.isocalendar().week
    df_features['sales_lag_1'] = df_features['sales'].shift(1)
    df_features['sales_lag_7'] = df_features['sales'].shift(7)
    df_features['ma_7'] = df_features['sales'].rolling(window=7, min_periods=1).mean()
    df_features['ma_30'] = df_features['sales'].rolling(window=30, min_periods=1).mean()
    df_features['deviation_ma_7'] = df_features['sales'] - df_features['ma_7']
    df_features['deviation_ma_30'] = df_features['sales'] - df_features['ma_30']
    df_features['ratio_ma_7'] = df_features['sales'] / df_features['ma_7']
    df_features['ratio_ma_30'] = df_features['sales'] / df_features['ma_30']
    return df_features

def detect_anomalies_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

def detect_anomalies_iqr(data, k=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return (data < lower_bound) | (data > upper_bound)

def evaluate_detection(df_clean):
    true_anomalies = df_clean['anomaly_type'] != 'normal'
    methods = ['anomaly_iso', 'anomaly_zscore', 'anomaly_iqr', 'anomaly_combined']
    results = {}
    for method in methods:
        if method == 'anomaly_iso':
            predicted = df_clean[method] == -1
        else:
            predicted = df_clean[method]
        tp = sum(true_anomalies & predicted)
        fp = sum(~true_anomalies & predicted)
        fn = sum(true_anomalies & ~predicted)
        tn = sum(~true_anomalies & ~predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results[method] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
    return results

def categorize_anomalies(df_anomalies):
    df_cat = df_anomalies.copy()
    median_sales = df_clean['sales'].median()
    conditions = [
        df_cat['sales'] > median_sales * 2,
        df_cat['sales'] < median_sales * 0.5,
        (df_cat['sales'] >= median_sales * 0.5) & (df_cat['sales'] <= median_sales * 2)
    ]
    choices = ['spike', 'drop', 'moderate']
    df_cat['anomaly_category'] = np.select(conditions, choices, default='unknown')
    return df_cat

def analyze_causes(df_anomalies):
    causes_analysis = {}
    weekend_anomalies = df_anomalies[df_anomalies['is_weekend'] == True]
    causes_analysis['weekend_effect'] = len(weekend_anomalies) / len(df_anomalies)
    seasonal_months = [11, 12, 1]
    seasonal_anomalies = df_anomalies[df_anomalies['month'].isin(seasonal_months)]
    causes_analysis['seasonal_effect'] = len(seasonal_anomalies) / len(df_anomalies)
    spikes = df_anomalies[df_anomalies['anomaly_category'] == 'spike']
    drops = df_anomalies[df_anomalies['anomaly_category'] == 'drop']
    causes_analysis['spike_ratio'] = len(spikes) / len(df_anomalies)
    causes_analysis['drop_ratio'] = len(drops) / len(df_anomalies)
    return causes_analysis

class AnomalyAlertSystem:
    def __init__(self, model, threshold_high=2.5, threshold_low=0.4):
        self.model = model
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.alerts = []

    def check_daily_sales(self, date, sales_value, expected_sales):
        ratio = sales_value / expected_sales
        alert_level = 'INFO'
        message = f"Ventes normales: {sales_value:.0f}€"

        if ratio > self.threshold_high:
            alert_level = 'HIGH'
            message = f"     SPIKE DÉTECTÉ: {sales_value:.0f}€ (x{ratio:.1f} vs attendu)"
        elif ratio < self.threshold_low:
            alert_level = 'CRITICAL'
            message = f" CHUTE DÉTECTÉE: {sales_value:.0f}€ ({ratio:.1%} vs attendu)"
        elif ratio > 1.5 or ratio < 0.7:
            alert_level = 'WARNING'
            message = f" VARIATION: {sales_value:.0f}€ ({ratio:.1f}x vs attendu)"

        alert = {
            'date': date,
            'sales': sales_value,
            'expected': expected_sales,
            'ratio': ratio,
            'level': alert_level,
            'message': message
        }
        self.alerts.append(alert)
        return alert

    def get_recommendations(self, alert):
        recommendations = []
        if alert['level'] == 'HIGH':
            recommendations.extend([
                "Vérifier les campagnes marketing actives",
                "Analyser les stocks pour éviter les ruptures",
                "Capitaliser sur cette tendance positive"
            ])
        elif alert['level'] == 'CRITICAL':
            recommendations.extend([
                "Vérifier les problèmes techniques",
                "Contrôler les stocks et approvisionnements",
                "Contacter l'équipe marketing pour des actions correctives"
            ])
        elif alert['level'] == 'WARNING':
            recommendations.extend([
                "Surveiller l'évolution sur les prochains jours",
                "Analyser les facteurs externes"
            ])
        return recommendations

def generate_executive_report(df_clean, anomalies, performance):
    #report = f"""RAPPORT EXÉCUTIF \n                        Détection d'Anomalies - Ventes \n                        RÉSUMÉ EXÉCUTIF \n                        • Période analysée: {df_clean['date'].min().strftime('%d/%m/%Y')} - \n                        {df_clean['date'].max().strftime('%d/%m/%Y')} \n                        • Ventes moyennes: {df_clean['sales'].mean():.0f}€/jour \n                        • Anomalies détectées: {len(anomalies)} ({len(anomalies)/len(df_clean)*100:.1f}% des jours) \n                        • Performance système: F1-Score = {performance['anomaly_combined']['f1_score']:.3f}    \n                        ANOMALIES PRINCIPALES \n                        • Spikes (hausses): {len(anomalies[anomalies['anomaly_category'] == 'spike'])} événements \n                        • Chutes: {len(anomalies[anomalies['anomaly_category'] == 'drop'])} événements \n                        • Impact moyen: {((anomalies['sales'] - df_clean['sales'].mean()) / df_clean['sales'].mean() * \n                        100).mean():.1f}% \n                        """
    report = f"""RAPPORT EXÉCUTIF \n Détection d'Anomalies - Ventes \n RÉSUMÉ EXÉCUTIF \n • Période analysée: {df_clean['date'].min().strftime('%d/%m/%Y')} - {df_clean['date'].max().strftime('%d/%m/%Y')} \n • Ventes moyennes: {df_clean['sales'].mean():.0f}€/jour \n • Anomalies détectées: {len(anomalies)} ({len(anomalies)/len(df_clean)*100:.1f}% des jours) \n • Performance système: F1-Score = {performance['anomaly_combined']['f1_score']:.3f} \n ANOMALIES PRINCIPALES \n • Spikes (hausses): {len(anomalies[anomalies['anomaly_category'] == 'spike'])} événements \n • Chutes: {len(anomalies[anomalies['anomaly_category'] == 'drop'])} événements \n • Impact moyen: {((anomalies['sales'] - df_clean['sales'].mean()) / df_clean['sales'].mean() * 100).mean():.1f}% \n """
    return report

@app.route('/')
def index():
    # Exécution du script d'analyse
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 6, 30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = generate_sales_data(date_range)
    df = inject_anomalies(df)
    df_features = create_features(df)

    feature_columns = ['sales', 'day_of_week', 'month', 'deviation_ma_7', 'deviation_ma_30', 'ratio_ma_7', 'ratio_ma_30']
    df_clean = df_features.dropna()
    X = df_clean[feature_columns]

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df_clean['anomaly_iso'] = iso_forest.fit_predict(X)
    df_clean['anomaly_zscore'] = detect_anomalies_zscore(df_clean['sales'])
    df_clean['anomaly_iqr'] = detect_anomalies_iqr(df_clean['sales'])
    df_clean['anomaly_combined'] = ((df_clean['anomaly_iso'] == -1) |
                                    df_clean['anomaly_zscore'] |
                                    df_clean['anomaly_iqr']
                                    )

    performance = evaluate_detection(df_clean)
    anomalies = df_clean[df_clean['anomaly_combined'] == True].copy()
    anomalies_cat = categorize_anomalies(anomalies)
    causes = analyze_causes(anomalies_cat)

    # Génération des graphiques
    img_buffers = {}
    plt.style.use('seaborn-v0_8')

    # Graphique 1: Série temporelle avec anomalies
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(df_clean['date'], df_clean['sales'], label='Ventes', alpha=0.7)
    axes[0, 0].scatter(anomalies['date'], anomalies['sales'],
                       color='red', s=50, label='Anomalies', zorder=5)
    axes[0, 0].set_title('Ventes avec Anomalies Détectées')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Ventes (€)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_buffers['plot1'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Graphique 2: Distribution des ventes
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df_clean['sales'], bins=50, alpha=0.7, label='Normal')
    ax.hist(anomalies['sales'], bins=20, alpha=0.7, color='red', label='Anomalies')
    ax.set_title('Distribution des Ventes')
    ax.set_xlabel('Ventes (€)')
    ax.set_ylabel('Fréquence')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_buffers['plot2'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Graphique 3: Anomalies par type
    fig, ax = plt.subplots(figsize=(7, 5))
    anomaly_counts = anomalies_cat['anomaly_category'].value_counts()
    ax.bar(anomaly_counts.index, anomaly_counts.values)
    ax.set_title('Anomalies par Catégorie')
    ax.set_ylabel('Nombre')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_buffers['plot3'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Graphique 4: Performance des algorithmes
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = list(performance.keys())
    f1_scores = [performance[method]['f1_score'] for method in methods]
    ax.bar(methods, f1_scores)
    ax.set_title('Performance des Algorithmes (F1-Score)')
    ax.set_ylabel('F1-Score')
    ax.tick_params(axis='x', rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_buffers['plot4'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # Génération du rapport exécutif
    executive_report = generate_executive_report(df_clean, anomalies_cat, performance)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport d'Anomalies de Ventes</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #0056b3; }}
            pre {{ background-color: #e2e2e2; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            img {{ max-width: 100%; height: auto; display: block; margin: 10px 0; border: 1px solid #ddd; }}
            .plot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Rapport d'Anomalies de Ventes</h1>
            <h2>Rapport Exécutif</h2>
            <pre>{executive_report}</pre>

            <h2>Statistiques des Données</h2>
            <pre>
                Période: {df['date'].min()} à {df['date'].max()}
                Nombre de jours: {len(df)}
                Ventes moyennes: {df['sales'].mean():.0f}€
                Écart-type: {df['sales'].std():.0f}€
            </pre>

            <h2>Anomalies Injectées</h2>
            <pre>{df['anomaly_type'].value_counts().to_string()}</pre>

            <h2>Performance des Algorithmes</h2>
            <pre>
                {'\n'.join([f"{method}: Précision={metrics['precision']:.3f}, Rappel={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}" for method, metrics in performance.items()])}
            </pre>

            <h2>Analyse des Anomalies Détectées</h2>
            <pre>
                Nombre d'anomalies: {len(anomalies)}
                Pourcentage des données: {len(anomalies) / len(df_clean) * 100:.1f}%

                Répartition par type réel:
{anomalies['anomaly_type'].value_counts().to_string()}

                Répartition par jour de la semaine:
{anomalies['day_of_week'].value_counts().sort_index().to_string()}

                Répartition par mois:
{anomalies['month'].value_counts().sort_index().to_string()}

                Catégorisation des anomalies:
{anomalies_cat['anomaly_category'].value_counts().to_string()}

                Analyse des causes:
                {'\n'.join([f"{cause}: {value:.1%}" for cause, value in causes.items()])}
            </pre>

            <h2>Graphiques</h2>
            <div class="plot-grid">
                <img src="data:image/png;base64,{img_buffers['plot1']}" alt="Ventes avec Anomalies Détectées">
                <img src="data:image/png;base64,{img_buffers['plot2']}" alt="Distribution des Ventes">
                <img src="data:image/png;base64,{img_buffers['plot3']}" alt="Anomalies par Catégorie">
                <img src="data:image/png;base64,{img_buffers['plot4']}" alt="Performance des Algorithmes">
            </div>

        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=27090)


