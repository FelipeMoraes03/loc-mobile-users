import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
import folium

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay



# DATA TREATMENT

# Split a dataframe into train, validation and test
def split_data(df : pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qtd_lines = df.shape[0]

    train = df.iloc[:int(qtd_lines * 0.7)]
    validation = df.iloc[int(qtd_lines * 0.7) : int(qtd_lines * 0.85)]
    test = df.iloc[int(qtd_lines * 0.85):]

    return train, validation, test

# Increases the number of instances of a dataframe by randomly replicating samples
def random_oversampling(num_samples : int, df : pd.DataFrame) -> pd.DataFrame:
    num_new_samples = num_samples - df.shape[0]

    df_samples = df.sample(n=num_new_samples, replace=True)
    df = pd.concat([df, df_samples], ignore_index=False)
    df = df.sample(frac=1, replace=False).reset_index(drop=True)
    
    return df

# Split a dataframe into train, validation and test and
# Balances the number of instances on train and validation
def split_and_balance(df0: pd.DataFrame, df1: pd.DataFrame) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df0_train, df0_val, df0_test = split_data(df0)
    df1_train, df1_val, df1_test = split_data(df1)

    if df0.shape[0] < df1.shape[0]:
        df0_train = random_oversampling(df1_train.shape[0], df0_train)
        df0_val = random_oversampling(df1_val.shape[0], df0_val)
    elif df0.shape[0] > df1.shape[0]:
        df1_train = random_oversampling(df0_train.shape[0], df1_train)
        df1_val = random_oversampling(df0_val.shape[0], df1_val)

    df_train = pd.concat([df0_train, df1_train], ignore_index=False)
    df_train = df_train.sample(frac=1, replace=False).reset_index(drop=True)

    df_val = pd.concat([df0_val, df1_val], ignore_index=False)
    df_val = df_val.sample(frac=1, replace=False).reset_index(drop=True)

    df_test = pd.concat([df0_test, df1_test], ignore_index=False)
    df_test = df_test.sample(frac=1, replace=False).reset_index(drop=True)

    return df_train, df_val, df_test



# MODEL EVALUATION

# Plot model's ROC curve
def plot_roc_curve(y_pred: np.array, y_test: np.array) -> None:
    
    # Obter as probabilidades da classe positiva
    y_pred_positive = y_pred[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_positive)
    roc_auc = roc_auc_score(y_test, y_pred_positive)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})' )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Plot model's confusion matrix
def plot_confusion_matrix(y_pred: np.array, y_test: np.array) -> None:
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Indoor', 'Outdoor'], yticklabels=['Indoor', 'Outdoor'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Computes model's metrics
def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.show()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics

# Prints model's metrics
def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))

# Converts one hot encoding to a string
def one_hot_to_string(df: pd.DataFrame, col: list) -> pd.DataFrame:
    for prefix in col:
        cols_to_concat = [c for c in df.columns if c.startswith(prefix + '_')]
        
        if cols_to_concat:
            df[prefix] = df[cols_to_concat].astype(str).agg(''.join, axis=1)
            
            df.drop(cols_to_concat, axis=1, inplace=True)
            
    return df



# FOLIUM
def plot_folium_map(y_test: np.array, y_pred: np.array, connect_point: bool = False, red_color : str = 'Pontos Preditos', blue_color : str ='Pontos Reais') -> folium.Map:
    # Criando o mapa centrado na primeira coordenada
    map = folium.Map(
        location=[-8.05, -34.95], 
        zoom_start=15
    )

    # Adicionando linhas conectando os pontos reais e preditos
    for real, pred in zip(y_test, y_pred):
        if connect_point:
            folium.PolyLine(locations=[real, pred], color='black', weight=1).add_to(map)

        folium.CircleMarker(
            location=[real[0], real[1]],
            radius=1,  # tamanho do ponto
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(map)

        folium.CircleMarker(
            location=[pred[0], pred[1]],
            radius=1,  # tamanho do ponto
            color='red',
            fill=True,
            fill_color='red'
        ).add_to(map)

    # Adicionando legenda no mapa
    legend_html = f'''
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 150px; height: 90px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white;
        ">
        &nbsp;<b>Legenda</b> <br>
        &nbsp;<i class="fa fa-circle" style="color:red"></i>&nbsp;{red_color}<br>
        &nbsp;<i class="fa fa-circle" style="color:blue"></i>&nbsp;{blue_color}
    </div>
    '''

    map.get_root().html.add_child(folium.Element(legend_html))

    # Exibindo o mapa
    return map

def calculate_accuracy(y_pred, y_true, threshold=0.0001):
    """
    Calculate the accuracy based on the Euclidean distance between real and estimated positions.

    Parameters:
    y_pred (array): Estimated positions.
    y_true (array): Real positions.
    threshold (float): Distance threshold to consider a prediction as accurate.

    Returns:
    float: Accuracy as the percentage of predictions within the threshold.
    """
    distances = np.linalg.norm(y_pred - y_true, axis=1)
    accurate_predictions = np.sum(distances <= threshold)
    accuracy = (accurate_predictions / len(distances))
    return accuracy


def distance_calc(coord1, coord2):
    # Radius of the Earth in meters
    R = 6371000
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in meters
    distance = R * c
    return distance


def get_error_distances(y_pred, y_true):
    return np.array([distance_calc(pred, true) for pred, true in zip(y_pred, y_true)])

def print_errors(distances : list):
    print(f"Erro de localização médio: {sum(distances) / len(distances):.3f} metros")
    print(f"Erro mínimo: {min(distances)} metros")
    print(f"Erro máximo: {max(distances):.3f} metros")
    print(f"Desvio Padrão do erro: {np.std(distances):.3f} metros")

def plot_boxplot(name: str, dist: list):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    box = ax.boxplot([dist], 
                     patch_artist=True,
                     labels=[name],
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     medianprops=dict(color='red'),
                     whiskerprops=dict(color='blue'),
                     capprops=dict(color='blue'),
                     flierprops=dict(markerfacecolor='blue', marker='o', markersize=5, linestyle='none', markeredgecolor='blue')
    )
    
    ax.set_title('BoxPlot do erro', fontsize=16)
    ax.set_ylabel('Erro em metros', fontsize=14)
    ax.set_xlabel('Modelo', fontsize=14)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.show()