import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

def load_data(verbose=0):
    with open('data_drops.pkl', 'rb') as file:
        data_loaded = pickle.load(file)

    # Access the variables
    segments = data_loaded['segments']
    s_label_data = data_loaded['s_label_data']
    v_label_data = data_loaded['v_label_data']
    unique_speeds = data_loaded['unique_speeds']
    unique_volumes = data_loaded['unique_volumes']
    speed_labels = np.searchsorted(unique_speeds, s_label_data)
    volume_labels = np.searchsorted(unique_volumes, v_label_data)
    del data_loaded
    if verbose > 0:
        print(f"Loaded segments shape: {segments.shape}")
        print(f"Speed labels shape: {speed_labels.shape}")
        print(f"Volume labels shape: {volume_labels.shape}")
        print(f"Unique speeds: {unique_speeds}")
        print(f"Unique volumes: {unique_volumes}")
    data = {
        'segments': segments,
        'speed_labels': speed_labels,
        'volume_labels': volume_labels,
        'unique_speeds': unique_speeds,
        'unique_volumes': unique_volumes
    }
    return data

def basic_feature_extractor(data,verbose=0):
    # Example feature extraction: mean and standard deviation of each segment
    segments = data['segments']
    stds = np.std(segments, axis=1)
    energy = np.sum(segments**2, axis=1)
    features = np.column_stack((energy, stds))
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    if verbose > 0:
        print(f"Extracted features shape: {features.shape}")
        print(f"First 5 features:\n{features[:5]}")
    # Normalize features (sklearn's StandardScaler)
    return features


def evaluate_classifier(classifier_speed, classifier_volume, X_speed, X_volume, data, random_state=42):
    """
    segments, speed_labels, volume_labels, unique_speeds, unique_volumes: are already loaded from the data_drops.pkl file.
    :param X_speed: scaled features, 2D numpy array of shape (n_samples, n_features)
    :param classifier: A scikit-learn classifier instance (e.g., RandomForestClassifier, SVC, etc.)
    :return:
    """
    speed_labels = data['speed_labels']
    volume_labels = data['volume_labels']

    # For speed classification (stratify by speed_labels)
    skf_speed = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    all_cv_scores_speed = []
    confusion_matrices_speed = []
    error_indices_speed = []

    for train_index, test_index in tqdm(skf_speed.split(X_speed, speed_labels),
                                        total=skf_speed.get_n_splits(X_speed, speed_labels)):
        X_train, X_test = X_speed[train_index], X_speed[test_index]
        y_train, y_test = speed_labels[train_index], speed_labels[test_index]

        classifier_speed.fit(X_train, y_train)
        y_pred = classifier_speed.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices_speed.append(cm)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Velocity accuracy (fold): {accuracy:.4f}')
        all_cv_scores_speed.append(accuracy)
        errors = test_index[y_test != y_pred]
        error_indices_speed.append(errors)

    print(f'Velocity accuracy average across all folds: {np.mean(all_cv_scores_speed):.4f}±{np.std(all_cv_scores_speed):.4f}')

    # For volume classification (stratify by volume_labels)
    skf_volume = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    all_cv_scores_volume = []
    confusion_matrices_volume = []
    error_indices_volume = []

    for train_index, test_index in tqdm(skf_volume.split(X_volume, volume_labels),
                                        total=skf_volume.get_n_splits(X_volume, volume_labels)):
        X_train, X_test = X_volume[train_index], X_volume[test_index]
        y_train, y_test = volume_labels[train_index], volume_labels[test_index]

        classifier_volume.fit(X_train, y_train)
        y_pred = classifier_volume.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices_volume.append(cm)
        accuracy = accuracy_score(y_test, y_pred)
        all_cv_scores_volume.append(accuracy)
        print(f'Volume accuracy (fold): {accuracy:.4f}')
        errors = test_index[y_test != y_pred]
        error_indices_volume.append(errors)

    print(f'Volume accuracy average across all folds - : {np.mean(all_cv_scores_volume):.4f}±{np.std(all_cv_scores_volume):.4f}')
    # Concatenate error indices from all folds
    error_indices_speed = np.concatenate(error_indices_speed)
    error_indices_volume = np.concatenate(error_indices_volume)
    return (confusion_matrices_speed, error_indices_speed, all_cv_scores_speed,
            confusion_matrices_volume, error_indices_volume, all_cv_scores_volume)

def directory_to_save(directory):
    """ Create a directory to save results if it does not exist."""
    """
    :param directory: Directory path where results will be saved.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_results(confusion_matrices_speed, error_indices_speed, all_cv_scores_speed,
                confusion_matrices_volume, error_indices_volume, all_cv_scores_volume,
                data,
                directory,verbose=0):
    unique_speeds = data['unique_speeds']
    unique_volumes = data['unique_volumes']

    directory_to_save(directory)
    directory = './' + directory + '/'
    # Save confusion matrices to CSV files
    cm_speed = np.sum(confusion_matrices_speed, axis=0)
    cm_speed_df = pd.DataFrame(cm_speed, index=unique_speeds, columns=unique_speeds)
    cm_speed_df.to_csv(directory + 'confusion_matrix_speed.csv')

    cm_vol = np.sum(confusion_matrices_volume, axis=0)
    cm_vol_df = pd.DataFrame(cm_vol, index=unique_volumes, columns=unique_volumes)
    cm_vol_df.to_csv(directory + 'confusion_matrix_volume.csv')

    # save validation and accuracy scores to CSV files
    speed_acc = np.trace(cm_speed)/ np.sum(cm_speed)
    cv_scores_speed_df = pd.DataFrame(all_cv_scores_speed, columns=['Speed Accuracy'])
    cv_scores_speed_df.loc['Mean'] = speed_acc
    cv_scores_speed_df.loc['Std'] = np.std(all_cv_scores_speed)
    cv_scores_speed_df.to_csv(directory + 'cv_scores_speed.csv')

    volume_acc = np.trace(cm_vol)/ np.sum(cm_vol)
    cv_scores_volume_df = pd.DataFrame(all_cv_scores_volume, columns=['Volume Accuracy'])
    cv_scores_volume_df.loc['Mean'] = volume_acc
    cv_scores_volume_df.loc['Std'] = np.std(all_cv_scores_volume)
    cv_scores_volume_df.to_csv(directory + 'cv_scores_volume.csv')

    # Define the data to be saved
    data_to_save = {
        'error_indices_speed': error_indices_speed,
        'error_indices_volume': error_indices_volume,
        'confusion_matrices_speed': confusion_matrices_speed,
        'confusion_matrices_volume': confusion_matrices_volume,
    }
    # Save the data to a MATLAB file
    with open(directory + 'error_info.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    if verbose > 0:
        print(f"Results saved in directory: {directory}")

def plot_results(confusion_matrices_speed, error_indices_speed,
                 confusion_matrices_volume, error_indices_volume,
                 data,
                 directory, verbose=0):
    unique_speeds = data['unique_speeds']
    unique_volumes = data['unique_volumes']
    speed_labels = data['speed_labels']
    volume_labels = data['volume_labels']

    directory_to_save(directory)
    directory = './' + directory + '/'
    # Calculate confusion matrices
    cm_speed = np.sum(confusion_matrices_speed, axis=0)
    cm_vol = np.sum(confusion_matrices_volume, axis=0)
    if verbose > 0:
        print(cm_speed)
        print(cm_vol)
    # Save confusion matrices to CSV files
    cm_speed_df = pd.DataFrame(cm_speed, index=unique_speeds, columns=unique_speeds)
    cm_vol_df = pd.DataFrame(cm_vol, index=unique_volumes, columns=unique_volumes)

    plt.figure(figsize=(3.5, 2.5))
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8
    sns.histplot(speed_labels[error_indices_volume].T,
                 discrete=True, legend=False,
                 linewidth=0.5, shrink=0.8)
    # Reduce the axes linewidth
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    plt.xlabel('Velocity (m/s)')
    plt.xticks(ticks=np.r_[0:7],
               labels=unique_speeds)
    #plt.yticks([0,6,12,25])
    plt.ylabel('Error counts')
    plt.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.5, color='black')
    plt.tight_layout()
    plt.savefig(directory + 'error_histogram_speed.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(directory + 'error_histogram_speed.jpeg', dpi=300)
    plt.show()

    plt.figure(figsize=(1.5, 1.5))
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8
    sns.histplot(volume_labels[error_indices_speed].T,
                 discrete=True, legend=False, shrink=0.8,
                 linewidth=0.5)
    # Reduce the axes linewidth
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    plt.xlabel('Volume (\u03BCL)')
    plt.xticks(ticks=np.arange(len(unique_volumes)),
               labels=unique_volumes, fontsize=10)
    # plt.yticks([1, 4])
    plt.ylabel('Error counts')
    plt.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.5, color='black')
    plt.tight_layout()
    plt.savefig(directory + 'error_histogram_volume.pdf', bbox_inches='tight')
    plt.savefig(directory + 'error_histogram_volume.jpeg', dpi=300)
    plt.show()

    # Convert DataFrames to numpy arrays
    cm_speed = cm_speed_df.to_numpy()
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10

    # Create a custom annotation matrix
    plt.figure(figsize=(3.5, 3.5))

    ax = sns.heatmap(cm_speed, annot=True, fmt='d',
                     cmap=ListedColormap(['aliceblue']), vmin=0, vmax=0,
                     square=True,
                     linewidths=0.5, linecolor='grey', linestyle = '--', # draw black grid lines
                     xticklabels=unique_speeds, yticklabels=unique_speeds,
                     cbar=False)
    for text in ax.texts:
        v = int(float(text.get_text()))
        if (v > 0) and (v < 50):
            text.set_weight('bold')

    sns.despine(left=False, right=False, top=False, bottom=False)
    plt.xlabel('Predicted velocity (m/s)')
    plt.ylabel('True velocity (m/s)')
    plt.tight_layout()
    # # Save the plot to PDF and JPEG
    plt.savefig(directory + 'confusion_matrix_speed.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(directory + 'confusion_matrix_speed.jpeg', dpi=300)
    plt.show()
    plt.figure(figsize=(2, 2.75))
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 10
    ax = sns.heatmap(cm_vol_df, annot=True, fmt='d',
                     cmap=ListedColormap(['aliceblue']), vmin=0, vmax=0,
                     square=True,
                     linewidths=0.5, linecolor='grey', linestyle='--',  # draw black grid lines
                     xticklabels=unique_volumes, yticklabels=unique_volumes,
                     cbar=False)
    for text in ax.texts:
        v = int(float(text.get_text()))
        text.set_fontsize(10)
        if (v > 0) and (v < 50):
            text.set_weight('bold')

    sns.despine(left=False, right=False, top=False, bottom=False)
    plt.xlabel(r'Predicted volume ($\mu$L)')
    plt.ylabel(r'True volume ($\mu$L)')
    plt.tight_layout()
    # # Save the plot to PDF and JPEG
    plt.savefig(directory + 'confusion_matrix_volume.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(directory + 'confusion_matrix_volume.jpeg', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    directory = 'test6'
    verbose = 1
    data = load_data(verbose=verbose)
    X = basic_feature_extractor(data, verbose=verbose)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    confusion_matrices_speed, error_indices_speed, all_cv_scores_speed, \
    confusion_matrices_volume, error_indices_volume, all_cv_scores_volume\
        = evaluate_classifier(classifier, classifier, X, data, random_state=42)
    save_results(confusion_matrices_speed, error_indices_speed, all_cv_scores_speed,
                 confusion_matrices_volume, error_indices_volume, all_cv_scores_volume,
                 data,
                 directory=directory, verbose=verbose)
    plot_results(confusion_matrices_speed, error_indices_speed,
                 confusion_matrices_volume, error_indices_volume,
                 data,
                 directory=directory, verbose=verbose)
