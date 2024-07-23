import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from EpilepsyFeatureExtractor import EpilepsyFeatureExtractor

extractor = EpilepsyFeatureExtractor()
extractor.load_data()
#Extract features to perform the classification.
stat_features = extractor.statistical_feature_extractor()
chaotic_features = extractor.chaotic_feature_extractor()

# Create labels: 0 for non-epileptic, 1 for epileptic
y = np.zeros(240)
y[160:] = 1  # The last 80 samples are epileptic

def run_classifiers(features, y):
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
        'SVM': svm.SVC(),
        'KNN': KNeighborsClassifier(n_neighbors=1)
    }

    results = []
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        pipe = Pipeline([('clf', clf)])
        scores = cross_val_score(pipe, features, y, cv=skf)
        results.append({
            'Classifier': name,
            'Mean Accuracy': np.mean(scores),
            'Std Dev': np.std(scores)
        })

    return results

# Run classifiers on statistical features
stat_results = run_classifiers(stat_features.T, y)

# Run classifiers on chaotic features
chaotic_results = run_classifiers(chaotic_features, y)

# Combine statistical and chaotic features
combined_features = np.hstack((stat_features.T, chaotic_features))

# Run classifiers on combined features
combined_results = run_classifiers(combined_features, y)

# Combine results and create a DataFrame
all_results = pd.DataFrame(stat_results + chaotic_results + combined_results)
all_results['Feature Type'] = ['Statistical']*len(stat_results) + ['Chaotic']*len(chaotic_results) + ['Combined']*len(combined_results)
# Display the results
print(all_results)
