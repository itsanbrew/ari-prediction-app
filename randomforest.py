import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

df = pd.read_csv('top12.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

print("\n" + "=" * 80)
print("FEATURE SELECTION")
print("=" * 80)

# Exclude disease-related columns (we only want symptoms)
columns_to_exclude = ['ARI?', 'diseases']  # ARI? is our target, diseases is excluded
feature_columns = [col for col in df.columns if col not in columns_to_exclude]

print(f"Total columns: {len(df.columns)}")
print(f"Excluded columns: {columns_to_exclude}")
print(f"Feature columns (symptoms): {len(feature_columns)}")

# Extract features (symptoms) and target
X = df[feature_columns].copy()
y = df['ARI?'].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:")
print(y.value_counts())
# Check for missing values
print(f"Missing values in features: {X.isnull().sum().sum()}")
print(f"Missing values in target: {y.isnull().sum()}")

# Check data types - ensure all are numeric
print(f"\nData types:")
print(X.dtypes.value_counts())

# ========== TUNING PARAMETERS ==========
# Adjust these parameters as needed:
TEST_SIZE = 0.2  # Proportion of data to use for testing (0.2 = 20%)
RANDOM_STATE_SPLIT = 42  # Random seed for reproducibility
SHUFFLE = True  # Whether to shuffle data before splitting
STRATIFY = True  # Whether to maintain class distribution in train/test splits
# ========================================

if STRATIFY:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE_SPLIT, 
        shuffle=SHUFFLE,
        stratify=y  # Maintains ARI/non-ARI distribution
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE_SPLIT, 
        shuffle=SHUFFLE
    )

print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"\nTraining set target distribution:")
print(y_train.value_counts())
print(f"\nOriginal test set target distribution:")
print(y_test.value_counts())

# ============================================================================
# CLASS WEIGHT CALCULATION - INVERSE PROPORTION
# ============================================================================
print("\n" + "=" * 80)
print("CALCULATING CLASS WEIGHTS (INVERSE PROPORTION)")
print("=" * 80)

# Calculate class distribution in training set (using original labels)
train_class_counts = y_train.value_counts()
total_train = len(y_train)

ari_count = train_class_counts.get('ARI', 0)
non_ari_count = train_class_counts.get('non-ARI', 0)

ari_proportion = ari_count / total_train
non_ari_proportion = non_ari_count / total_train

# Calculate inverse proportion weights
# Weight = 1 / proportion (normalized so they sum to number of classes)
ari_weight = 1.0 / ari_proportion
non_ari_weight = 1.0 / non_ari_proportion

# Normalize weights (optional - sklearn handles this, but we can show both)
# Normalized weights sum to number of classes
normalized_ari_weight = ari_weight / (ari_weight + non_ari_weight) * 2
normalized_non_ari_weight = non_ari_weight / (ari_weight + non_ari_weight) * 2

print(f"\nClass distribution in training set:")
print(f"  ARI: {ari_count} ({ari_proportion*100:.2f}%)")
print(f"  non-ARI: {non_ari_count} ({non_ari_proportion*100:.2f}%)")

print(f"\nInverse proportion weights (raw):")
print(f"  ARI weight: {ari_weight:.4f}")
print(f"  non-ARI weight: {non_ari_weight:.4f}")

print(f"\nInverse proportion weights (normalized):")
print(f"  ARI weight: {normalized_ari_weight:.4f}")
print(f"  non-ARI weight: {normalized_non_ari_weight:.4f}")

# Create class weight dictionary
CLASS_WEIGHT_DICT = {
    'ARI': ari_weight,
    'non-ARI': non_ari_weight
}

print(f"\nUsing class weights: {CLASS_WEIGHT_DICT}")

# ============================================================================
# WEIGHTED TEST SET CREATION - TUNING PARAMETERS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING WEIGHTED TEST SET")
print("=" * 80)

# ========== TUNING PARAMETERS ==========
USE_WEIGHTED_TEST_SET = True  # Whether to use weighted test set
ARI_PERCENTAGE = 0.5  # Percentage of ARI cases in weighted test set (0.5 = 50%)
RANDOM_STATE_WEIGHTED = 42  # Random seed for weighted test set sampling
# ========================================

if USE_WEIGHTED_TEST_SET:
    # Separate ARI and non-ARI cases in the original test set
    test_ari = X_test[y_test == 'ARI']
    test_non_ari = X_test[y_test == 'non-ARI']
    y_test_ari = y_test[y_test == 'ARI']
    y_test_non_ari = y_test[y_test == 'non-ARI']
    
    # Calculate how many samples we need for 50/50 split
    # We'll use the minimum of available ARI and non-ARI cases to ensure balanced split
    n_ari_available = len(test_ari)
    n_non_ari_available = len(test_non_ari)
    
    # Use the smaller of the two to ensure we can create a balanced set
    n_samples_per_class = min(n_ari_available, n_non_ari_available)
    
    # Sample equal numbers from both classes
    test_ari_sampled = test_ari.sample(n=n_samples_per_class, random_state=RANDOM_STATE_WEIGHTED)
    test_non_ari_sampled = test_non_ari.sample(n=n_samples_per_class, random_state=RANDOM_STATE_WEIGHTED)
    y_test_ari_sampled = y_test_ari[test_ari_sampled.index]
    y_test_non_ari_sampled = y_test_non_ari[test_non_ari_sampled.index]
    
    # Update variables for combination
    test_ari = test_ari_sampled
    test_non_ari = test_non_ari_sampled
    y_test_ari = y_test_ari_sampled
    y_test_non_ari = y_test_non_ari_sampled
    
    # Combine to create weighted test set
    X_test_weighted = pd.concat([test_ari, test_non_ari], axis=0).reset_index(drop=True)
    y_test_weighted = pd.concat([y_test_ari, y_test_non_ari], axis=0).reset_index(drop=True)
    
    # Shuffle the weighted test set
    np.random.seed(RANDOM_STATE_WEIGHTED)
    shuffle_indices = np.random.permutation(len(X_test_weighted))
    X_test_weighted = X_test_weighted.iloc[shuffle_indices].reset_index(drop=True)
    y_test_weighted = y_test_weighted.iloc[shuffle_indices].reset_index(drop=True)
    
    # Replace original test set with weighted test set
    X_test = X_test_weighted
    y_test = y_test_weighted
    
    print(f"Weighted test set created:")
    print(f"  Total samples: {len(X_test)}")
    print(f"  Target split: {ARI_PERCENTAGE * 100}% ARI, {(1-ARI_PERCENTAGE) * 100}% non-ARI")
    print(f"\nWeighted test set target distribution:")
    print(y_test.value_counts())
    print(f"\nActual split in weighted test set:")
    print(f"  ARI: {y_test.value_counts().get('ARI', 0)} ({y_test.value_counts().get('ARI', 0) / len(y_test) * 100:.2f}%)")
    print(f"  non-ARI: {y_test.value_counts().get('non-ARI', 0)} ({y_test.value_counts().get('non-ARI', 0) / len(y_test) * 100:.2f}%)")
else:
    print("Using original test set (not weighted)")

# ============================================================================
# RANDOM FOREST MODEL - TUNING PARAMETERS
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 80)

# ========== TUNING PARAMETERS ==========
# Adjust these Random Forest hyperparameters as needed:
N_ESTIMATORS = 100  # Number of trees in the forest
MAX_DEPTH = None  # Maximum depth of trees (None = no limit)
MIN_SAMPLES_SPLIT = 2  # Minimum samples required to split a node
MIN_SAMPLES_LEAF = 1  # Minimum samples required at a leaf node
MAX_FEATURES = 'sqrt'  # Number of features to consider for best split ('sqrt', 'log2', None, or int)
RANDOM_STATE_MODEL = 42  # Random seed for model reproducibility
N_JOBS = -1  # Number of parallel jobs (-1 = use all processors)
USE_CLASS_WEIGHTS = True  # Whether to use inverse proportion class weights (calculated above)
# ========================================

# Set class_weight for Random Forest
if USE_CLASS_WEIGHTS:
    CLASS_WEIGHT = CLASS_WEIGHT_DICT
else:
    CLASS_WEIGHT = None

rf_model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    max_features=MAX_FEATURES,
    random_state=RANDOM_STATE_MODEL,
    n_jobs=N_JOBS,
    class_weight=CLASS_WEIGHT
)

print("Training model...")
print(f"Model parameters:")
print(f"  n_estimators: {N_ESTIMATORS}")
print(f"  max_depth: {MAX_DEPTH}")
print(f"  min_samples_split: {MIN_SAMPLES_SPLIT}")
print(f"  min_samples_leaf: {MIN_SAMPLES_LEAF}")
print(f"  max_features: {MAX_FEATURES}")
print(f"  class_weight: {CLASS_WEIGHT}")

rf_model.fit(X_train, y_train)
print("Model training completed!")

# ============================================================================
# SAVE MODEL FOR DEPLOYMENT
# ============================================================================
print("\n" + "=" * 80)
print("SAVING MODEL FOR DEPLOYMENT")
print("=" * 80)

# ========== TUNING PARAMETERS ==========
SAVE_MODEL = True  # Whether to save the trained model
MODEL_FILENAME = 'ari_prediction_model.pkl'  # Filename for saved model
METADATA_FILENAME = 'model_metadata.json'  # Filename for model metadata
# ========================================

if SAVE_MODEL:
    # Save the trained model
    joblib.dump(rf_model, MODEL_FILENAME)
    print(f"Model saved to: {MODEL_FILENAME}")
    
    # Save metadata (feature columns, class weights, etc.)
    metadata = {
        'feature_columns': feature_columns,
        'model_type': 'RandomForestClassifier',
        'n_estimators': N_ESTIMATORS,
        'max_depth': str(MAX_DEPTH),
        'min_samples_split': MIN_SAMPLES_SPLIT,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'max_features': str(MAX_FEATURES),
        'class_weight': str(CLASS_WEIGHT),
        'target_classes': ['ARI', 'non-ARI'],
        'n_features': len(feature_columns),
        'training_samples': len(X_train)
    }
    
    with open(METADATA_FILENAME, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model metadata saved to: {METADATA_FILENAME}")
    print(f"\nModel can be loaded later with:")
    print(f"  import joblib")
    print(f"  model = joblib.load('{MODEL_FILENAME}')")
else:
    print("Model saving disabled (SAVE_MODEL = False)")

# ============================================================================
# MODEL PREDICTION
# ============================================================================
print("\n" + "=" * 80)
print("MAKING PREDICTIONS")
print("=" * 80)

y_test_pred = rf_model.predict(X_test)

# Get prediction probabilities for ROC curve
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL EVALUATION (TEST SET)")
print("=" * 80)

# Test set metrics
print("\n--- TEST SET METRICS ---")
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {test_accuracy:.4f}")

test_auc = roc_auc_score(y_test, y_test_proba)
print(f"ROC-AUC Score: {test_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['ARI', 'non-ARI']))

# Confusion Matrix
print("\n--- CONFUSION MATRIX (TEST SET) ---")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print("\nConfusion Matrix (Normalized):")
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# ========== TUNING PARAMETERS ==========
TOP_N_FEATURES = 20  # Number of top features to display
# ========================================

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop {TOP_N_FEATURES} Most Important Features:")
print(feature_importance.head(TOP_N_FEATURES).to_string(index=False))

# ============================================================================
# VISUALIZATION (OPTIONAL)
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# ========== TUNING PARAMETERS ==========
SAVE_PLOTS = True  # Whether to save plots to files
PLOT_FORMAT = 'png'  # Format for saved plots ('png', 'pdf', 'svg')
SHOW_PLOTS = False  # Whether to display plots (set to True to see plots)
# ========================================

try:
    
    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ARI', 'non-ARI'], 
                yticklabels=['ARI', 'non-ARI'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Saved: confusion_matrix.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
    # ROC Curve
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba, pos_label='ARI')
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {test_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_PLOTS:
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("Saved: roc_curve.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    
except Exception as e:
    print(f"Warning: Could not generate plots: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)
print(f"\nModel: Random Forest Classifier")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features used: {len(feature_columns)}")
print(f"\nTest Set Performance:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  ROC-AUC: {test_auc:.4f}")
print(f"\nTraining completed successfully!")

