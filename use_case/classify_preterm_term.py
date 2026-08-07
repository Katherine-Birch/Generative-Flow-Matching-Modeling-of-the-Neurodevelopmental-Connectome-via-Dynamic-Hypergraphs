
import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold



def load_and_vectorize_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    matrices = np.array([d['matrix'] for d in data])
    ages = np.array([d['age'] for d in data])

    triu_indices = np.triu_indices(90, k=1)

    vectors = []
    for mat in matrices:
        mat = np.log1p(mat)
        vectors.append(mat[triu_indices])

    labels = (ages >= 37).astype(int)

    return np.array(vectors), labels, ages


def fuzzy_label(ga, cutoff, temp):
    return 1 / (1 + np.exp(-(ga - cutoff) / temp))


def run_ratio_sweep_cv_bootstrapped(
        train_path,
        test_path,
        synthetic_path,
        n_splits=5,
        n_bootstrap=100,
        random_seed=42):
    rng = np.random.default_rng(random_seed)

    print("Loading datasets...")

    X_train_full, y_train_full, train_ages_full = load_and_vectorize_pkl(train_path)
    X_test, y_test, _ = load_and_vectorize_pkl(test_path)
    X_syn, y_syn, syn_ages = load_and_vectorize_pkl(synthetic_path)

    ratios = [0.0, 0.25, 0.5, 1.0, 2.0]
    results = []

    print("\n" + "=" * 70)
    print(f"{n_splits}-FOLD CV + BOOTSTRAPPED AUGMENTATION SWEEP")
    print("Train: Real (K-Fold Subsets) + Synthetic")
    print("Test : Real")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    for ratio in ratios:
        metric_storage = {
            "Preterm_P": [], "Preterm_R": [], "Preterm_F1": [],
            "Term_P": [], "Term_R": [], "Term_F1": [],
            "Accuracy": [], "ROC_AUC": []
        }

        print(f"\nRunning ratio {ratio}...")

        # --- OUTER LOOP: 5-Fold CV on Real Training Data ---
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):

            X_fold_train = X_train_full[train_idx]
            y_fold_train = y_train_full[train_idx]
            ages_fold_train = train_ages_full[train_idx]

            # --- INNER LOOP: Bootstrapping ---
            for trial in range(n_bootstrap):

                # 1. BUILD TRAINING SET
                if ratio == 0.0:
                    X_combined = X_fold_train
                    y_combined = y_fold_train
                    ages_combined = ages_fold_train
                else:
                    n_syn = int(len(X_fold_train) * ratio)
                    syn_idx = rng.choice(len(X_syn), size=n_syn, replace=(n_syn > len(X_syn)))

                    X_syn_subset = X_syn[syn_idx]
                    y_syn_subset = y_syn[syn_idx]
                    age_syn_subset = syn_ages[syn_idx]

                    X_combined = np.vstack((X_fold_train, X_syn_subset))
                    y_combined = np.concatenate((y_fold_train, y_syn_subset))
                    ages_combined = np.concatenate((ages_fold_train, age_syn_subset))

                # 2. SCALE
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_combined)

                # 3. FUZZY WEIGHTS
                soft_labels = fuzzy_label(ages_combined, cutoff=37, temp=1.0)
                weights = np.abs(soft_labels - 0.5) * 2

                # 4. TRAIN MODEL
                model = LogisticRegression(max_iter=1000, solver='lbfgs')
                model.fit(X_train_scaled, y_combined, sample_weight=weights)

                # 5. BOOTSTRAP REAL TEST SET
                test_idx = rng.choice(len(X_test), size=len(X_test), replace=True)
                X_test_boot = X_test[test_idx]
                y_test_boot = y_test[test_idx]

                X_test_boot_scaled = scaler.transform(X_test_boot)

                # 6. EVALUATE
                y_pred = model.predict(X_test_boot_scaled)
                y_prob = model.predict_proba(X_test_boot_scaled)[:, 1]

                report = classification_report(
                    y_test_boot, y_pred, target_names=["Preterm", "Term"],
                    output_dict=True, zero_division=0
                )

                metric_storage["Preterm_P"].append(report["Preterm"]["precision"])
                metric_storage["Preterm_R"].append(report["Preterm"]["recall"])
                metric_storage["Preterm_F1"].append(report["Preterm"]["f1-score"])
                metric_storage["Term_P"].append(report["Term"]["precision"])
                metric_storage["Term_R"].append(report["Term"]["recall"])
                metric_storage["Term_F1"].append(report["Term"]["f1-score"])
                metric_storage["Accuracy"].append(balanced_accuracy_score(y_test_boot, y_pred))
                metric_storage["ROC_AUC"].append(roc_auc_score(y_test_boot, y_prob))

        # Summarize across all 500 evaluations (5 folds * 100 bootstraps)
        summary = {}
        for metric, values in metric_storage.items():
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "ci_low": np.percentile(values, 2.5),
                "ci_high": np.percentile(values, 97.5)
            }

        results.append({
            "Ratio": ratio,
            "Preterm_P": f"{summary['Preterm_P']['mean']:.3f} ± {summary['Preterm_P']['std']:.3f}",
            "Preterm_R": f"{summary['Preterm_R']['mean']:.3f} ± {summary['Preterm_R']['std']:.3f}",
            "Preterm_F1": f"{summary['Preterm_F1']['mean']:.3f} ± {summary['Preterm_F1']['std']:.3f}",
            "Term_P": f"{summary['Term_P']['mean']:.3f} ± {summary['Term_P']['std']:.3f}",
            "Term_R": f"{summary['Term_R']['mean']:.3f} ± {summary['Term_R']['std']:.3f}",
            "Term_F1": f"{summary['Term_F1']['mean']:.3f} ± {summary['Term_F1']['std']:.3f}",
            "Accuracy": f"{summary['Accuracy']['mean']:.3f} ± {summary['Accuracy']['std']:.3f}",
            "ROC_AUC": f"{summary['ROC_AUC']['mean']:.3f} ± {summary['ROC_AUC']['std']:.3f}"
        })

    import pandas as pd
    df = pd.DataFrame(results)
    print("\n" + "=" * 140)
    print("FINAL CV + BOOTSTRAPPED RESULTS")
    print("=" * 140)
    print(df.to_string(index=False))
    df.to_csv("augmentation_cv_bootstrap_results.csv", index=False)
    return df


def run_ratio_sweep_bootstrapped(
        train_path,
        test_path,
        synthetic_path,
        n_bootstrap=100,
        random_seed=42):

    rng = np.random.default_rng(random_seed)

    print("Loading datasets...")

    X_train, y_train, train_ages = load_and_vectorize_pkl(train_path)
    X_test, y_test, _ = load_and_vectorize_pkl(test_path)
    X_syn, y_syn, syn_ages = load_and_vectorize_pkl(synthetic_path)

    ratios = [0.0, 0.25, 0.5, 1.0, 2.0]

    results = []

    print("\n" + "=" * 70)
    print("BOOTSTRAPPED AUGMENTATION SWEEP")
    print("Train: Real + Synthetic")
    print("Test : Real")
    print("=" * 70)

    for ratio in ratios:

        metric_storage = {
            "Preterm_P": [],
            "Preterm_R": [],
            "Preterm_F1": [],
            "Term_P": [],
            "Term_R": [],
            "Term_F1": [],
            "Accuracy": [],
            "ROC_AUC": []
        }

        print(f"\nRunning ratio {ratio}...")

        for trial in range(n_bootstrap):

            # --------------------------------------------------
            # BUILD TRAINING SET
            # --------------------------------------------------

            if ratio == 0.0:

                X_combined = X_train
                y_combined = y_train
                ages_combined = train_ages

            else:

                n_syn = int(len(X_train) * ratio)

                syn_idx = rng.choice(
                    len(X_syn),
                    size=n_syn,
                    replace=(n_syn > len(X_syn))
                )

                X_syn_subset = X_syn[syn_idx]
                y_syn_subset = y_syn[syn_idx]
                age_syn_subset = syn_ages[syn_idx]

                X_combined = np.vstack((X_train, X_syn_subset))
                y_combined = np.concatenate((y_train, y_syn_subset))
                ages_combined = np.concatenate(
                    (train_ages, age_syn_subset)
                )

            # --------------------------------------------------
            # SCALE
            # --------------------------------------------------

            scaler = StandardScaler()

            X_train_scaled = scaler.fit_transform(X_combined)

            # --------------------------------------------------
            # FUZZY WEIGHTS
            # --------------------------------------------------

            soft_labels = fuzzy_label(
                ages_combined,
                cutoff=37,
                temp=1.0
            )

            weights = np.abs(soft_labels - 0.5) * 2

            # --------------------------------------------------
            # TRAIN MODEL
            # --------------------------------------------------

            model = LogisticRegression(
                max_iter=1000,
                solver='lbfgs'
            )

            model.fit(
                X_train_scaled,
                y_combined,
                sample_weight=weights
            )

            # --------------------------------------------------
            # BOOTSTRAP REAL TEST SET
            # --------------------------------------------------

            test_idx = rng.choice(
                len(X_test),
                size=len(X_test),
                replace=True
            )

            X_test_boot = X_test[test_idx]
            y_test_boot = y_test[test_idx]

            X_test_boot_scaled = scaler.transform(X_test_boot)

            # --------------------------------------------------
            # EVALUATE
            # --------------------------------------------------

            y_pred = model.predict(X_test_boot_scaled)
            y_prob = model.predict_proba(X_test_boot_scaled)[:, 1]

            report = classification_report(
                y_test_boot,
                y_pred,
                target_names=["Preterm", "Term"],
                output_dict=True,
                zero_division=0
            )

            metric_storage["Preterm_P"].append(
                report["Preterm"]["precision"]
            )

            metric_storage["Preterm_R"].append(
                report["Preterm"]["recall"]
            )

            metric_storage["Preterm_F1"].append(
                report["Preterm"]["f1-score"]
            )

            metric_storage["Term_P"].append(
                report["Term"]["precision"]
            )

            metric_storage["Term_R"].append(
                report["Term"]["recall"]
            )

            metric_storage["Term_F1"].append(
                report["Term"]["f1-score"]
            )

            metric_storage["Accuracy"].append(
                balanced_accuracy_score(y_test_boot, y_pred)
            )

            metric_storage["ROC_AUC"].append(
                roc_auc_score(y_test_boot, y_prob)
            )

        summary = {}

        for metric, values in metric_storage.items():
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "ci_low": np.percentile(values, 2.5),
                "ci_high": np.percentile(values, 97.5)
            }

        results.append({
            "Ratio": ratio,

            "Preterm_P":
                f"{summary['Preterm_P']['mean']:.3f} ± {summary['Preterm_P']['std']:.3f}",

            "Preterm_R":
                f"{summary['Preterm_R']['mean']:.3f} ± {summary['Preterm_R']['std']:.3f}",

            "Preterm_F1":
                f"{summary['Preterm_F1']['mean']:.3f} ± {summary['Preterm_F1']['std']:.3f}",

            "Term_P":
                f"{summary['Term_P']['mean']:.3f} ± {summary['Term_P']['std']:.3f}",

            "Term_R":
                f"{summary['Term_R']['mean']:.3f} ± {summary['Term_R']['std']:.3f}",

            "Term_F1":
                f"{summary['Term_F1']['mean']:.3f} ± {summary['Term_F1']['std']:.3f}",

            "Accuracy":
                f"{summary['Accuracy']['mean']:.3f} ± {summary['Accuracy']['std']:.3f}",

            "ROC_AUC":
                f"{summary['ROC_AUC']['mean']:.3f} ± {summary['ROC_AUC']['std']:.3f}"
        })





    # ======================================================
    # PLOT WITH ERROR BARS
    # ======================================================
    #
    # ratios = [r["ratio"] for r in results]
    #
    # roc_means = [r["roc_mean"] for r in results]
    # roc_stds = [r["roc_std"] for r in results]
    #
    # recall_means = [r["recall_mean"] for r in results]
    # recall_stds = [r["recall_std"] for r in results]

    # fig, (ax1, ax2) = plt.subplots(
    #     1,
    #     2,
    #     figsize=(12, 5)
    # )
    #
    # ax1.errorbar(
    #     ratios,
    #     roc_means,
    #     yerr=roc_stds,
    #     marker='o',
    #     capsize=5
    # )
    #
    # ax1.set_title("ROC-AUC")
    # ax1.set_xlabel("Synthetic / Real Ratio")
    # ax1.set_ylabel("ROC-AUC")
    # ax1.grid(True)
    #
    # ax2.errorbar(
    #     ratios,
    #     recall_means,
    #     yerr=recall_stds,
    #     marker='s',
    #     capsize=5
    # )
    #
    # ax2.set_title("Preterm Recall")
    # ax2.set_xlabel("Synthetic / Real Ratio")
    # ax2.set_ylabel("Recall")
    # ax2.grid(True)
    #
    # plt.suptitle(
    #     "Effect of Synthetic Data Augmentation\n"
    #     "(Mean ± SD across bootstrap trials)"
    # )
    #
    # plt.tight_layout()
    # plt.show()

    import pandas as pd

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 140)
    print("FINAL BOOTSTRAPPED RESULTS")
    print("=" * 140)

    print(df.to_string(index=False))

    df.to_csv(
        "augmentation_bootstrap_results.csv",
        index=False
    )

    return df


def run_real_train_synthetic_test(
        train_path,
        synthetic_path,
        n_bootstrap=100,
        random_seed=42):
    rng = np.random.default_rng(random_seed)

    print("Loading datasets...")
    X_train, y_train, train_ages = load_and_vectorize_pkl(train_path)
    X_syn, y_syn, syn_ages = load_and_vectorize_pkl(synthetic_path)

    print("\n" + "=" * 70)
    print("EXPERIMENT: TRAIN ON REAL, BOOTSTRAP TEST ON SYNTHETIC")
    print("Train: Real")
    print("Test : Synthetic")
    print("=" * 70)

    # --------------------------------------------------
    # 1. BUILD & SCALE TRAINING SET (Done ONCE)
    # --------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply your fuzzy weights to the real training data
    soft_labels = fuzzy_label(train_ages, cutoff=37, temp=1.0)
    weights = np.abs(soft_labels - 0.5) * 2

    # Train the model once
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train, sample_weight=weights)

    # --------------------------------------------------
    # 2. BOOTSTRAP SYNTHETIC TEST SET
    # --------------------------------------------------
    metric_storage = {
        "Preterm_P": [], "Preterm_R": [], "Preterm_F1": [],
        "Term_P": [], "Term_R": [], "Term_F1": [],
        "Accuracy": [], "ROC_AUC": []
    }

    print(f"\nRunning {n_bootstrap} bootstrap trials on synthetic test set...")

    for trial in range(n_bootstrap):
        # Sample synthetic data with replacement
        test_idx = rng.choice(len(X_syn), size=len(X_syn), replace=True)
        X_syn_boot = X_syn[test_idx]
        y_syn_boot = y_syn[test_idx]

        # Transform synthetic test set using the REAL data's scaler
        X_syn_boot_scaled = scaler.transform(X_syn_boot)

        # Evaluate
        y_pred = model.predict(X_syn_boot_scaled)
        y_prob = model.predict_proba(X_syn_boot_scaled)[:, 1]

        report = classification_report(
            y_syn_boot, y_pred, target_names=["Preterm", "Term"],
            output_dict=True, zero_division=0
        )

        metric_storage["Preterm_P"].append(report["Preterm"]["precision"])
        metric_storage["Preterm_R"].append(report["Preterm"]["recall"])
        metric_storage["Preterm_F1"].append(report["Preterm"]["f1-score"])

        metric_storage["Term_P"].append(report["Term"]["precision"])
        metric_storage["Term_R"].append(report["Term"]["recall"])
        metric_storage["Term_F1"].append(report["Term"]["f1-score"])

        metric_storage["Accuracy"].append(balanced_accuracy_score(y_syn_boot, y_pred))
        metric_storage["ROC_AUC"].append(roc_auc_score(y_syn_boot, y_prob))

    # --------------------------------------------------
    # 3. SUMMARIZE AND EXPORT
    # --------------------------------------------------
    summary = {}
    for metric, values in metric_storage.items():
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "ci_low": np.percentile(values, 2.5),
            "ci_high": np.percentile(values, 97.5)
        }

    # Store results in a single-row format matching your style
    results = [{
        "Experiment": "Real_Train_Syn_Test",
        "Preterm_P": f"{summary['Preterm_P']['mean']:.3f} ± {summary['Preterm_P']['std']:.3f}",
        "Preterm_R": f"{summary['Preterm_R']['mean']:.3f} ± {summary['Preterm_R']['std']:.3f}",
        "Preterm_F1": f"{summary['Preterm_F1']['mean']:.3f} ± {summary['Preterm_F1']['std']:.3f}",
        "Term_P": f"{summary['Term_P']['mean']:.3f} ± {summary['Term_P']['std']:.3f}",
        "Term_R": f"{summary['Term_R']['mean']:.3f} ± {summary['Term_R']['std']:.3f}",
        "Term_F1": f"{summary['Term_F1']['mean']:.3f} ± {summary['Term_F1']['std']:.3f}",
        "Accuracy": f"{summary['Accuracy']['mean']:.3f} ± {summary['Accuracy']['std']:.3f}",
        "ROC_AUC": f"{summary['ROC_AUC']['mean']:.3f} ± {summary['ROC_AUC']['std']:.3f}"
    }]

    import pandas as pd
    df = pd.DataFrame(results)

    print("\n")
    print("=" * 140)
    print("FINAL BOOTSTRAPPED RESULTS (REAL TRAIN -> SYNTHETIC TEST)")
    print("=" * 140)
    print(df.to_string(index=False))

    df.to_csv("real_train_synthetic_test_bootstrap_results.csv", index=False)

    return df


def run_real_train_synthetic_test_cv(
        train_path,
        synthetic_path,
        n_splits=5,
        n_bootstrap=100,
        random_seed=42):
    rng = np.random.default_rng(random_seed)

    print("Loading datasets...")
    X_train_full, y_train_full, train_ages_full = load_and_vectorize_pkl(train_path)
    X_syn, y_syn, syn_ages = load_and_vectorize_pkl(synthetic_path)

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {n_splits}-FOLD CV ON REAL TRAIN, BOOTSTRAP SYNTHETIC TEST")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    metric_storage = {
        "Preterm_P": [], "Preterm_R": [], "Preterm_F1": [],
        "Term_P": [], "Term_R": [], "Term_F1": [],
        "Accuracy": [], "ROC_AUC": []
    }

    # --- OUTER LOOP: 5-Fold CV on Real Training Data ---
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"Training and Evaluating Fold {fold + 1}/{n_splits}...")

        X_fold_train = X_train_full[train_idx]
        y_fold_train = y_train_full[train_idx]
        ages_fold_train = train_ages_full[train_idx]

        # 1. BUILD & SCALE TRAINING SET (Once per fold)
        scaler = StandardScaler()
        X_fold_train_scaled = scaler.fit_transform(X_fold_train)

        # Apply fuzzy weights
        soft_labels = fuzzy_label(ages_fold_train, cutoff=37, temp=1.0)
        weights = np.abs(soft_labels - 0.5) * 2

        # Train model for this fold
        model = LogisticRegression(max_iter=1000, solver='lbfgs')
        model.fit(X_fold_train_scaled, y_fold_train, sample_weight=weights)

        # --- INNER LOOP: BOOTSTRAP SYNTHETIC TEST SET ---
        for trial in range(n_bootstrap):
            # Sample synthetic data with replacement
            test_idx = rng.choice(len(X_syn), size=len(X_syn), replace=True)
            X_syn_boot = X_syn[test_idx]
            y_syn_boot = y_syn[test_idx]

            # Transform synthetic test set using THIS FOLD'S scaler
            X_syn_boot_scaled = scaler.transform(X_syn_boot)

            # Evaluate
            y_pred = model.predict(X_syn_boot_scaled)
            y_prob = model.predict_proba(X_syn_boot_scaled)[:, 1]

            report = classification_report(
                y_syn_boot, y_pred, target_names=["Preterm", "Term"],
                output_dict=True, zero_division=0
            )

            metric_storage["Preterm_P"].append(report["Preterm"]["precision"])
            metric_storage["Preterm_R"].append(report["Preterm"]["recall"])
            metric_storage["Preterm_F1"].append(report["Preterm"]["f1-score"])
            metric_storage["Term_P"].append(report["Term"]["precision"])
            metric_storage["Term_R"].append(report["Term"]["recall"])
            metric_storage["Term_F1"].append(report["Term"]["f1-score"])
            metric_storage["Accuracy"].append(balanced_accuracy_score(y_syn_boot, y_pred))
            metric_storage["ROC_AUC"].append(roc_auc_score(y_syn_boot, y_prob))

    # Summarize across all 500 evaluations
    summary = {}
    for metric, values in metric_storage.items():
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "ci_low": np.percentile(values, 2.5),
            "ci_high": np.percentile(values, 97.5)
        }

    results = [{
        "Experiment": "Real_Train_Syn_Test_CV",
        "Preterm_P": f"{summary['Preterm_P']['mean']:.3f} ± {summary['Preterm_P']['std']:.3f}",
        "Preterm_R": f"{summary['Preterm_R']['mean']:.3f} ± {summary['Preterm_R']['std']:.3f}",
        "Preterm_F1": f"{summary['Preterm_F1']['mean']:.3f} ± {summary['Preterm_F1']['std']:.3f}",
        "Term_P": f"{summary['Term_P']['mean']:.3f} ± {summary['Term_P']['std']:.3f}",
        "Term_R": f"{summary['Term_R']['mean']:.3f} ± {summary['Term_R']['std']:.3f}",
        "Term_F1": f"{summary['Term_F1']['mean']:.3f} ± {summary['Term_F1']['std']:.3f}",
        "Accuracy": f"{summary['Accuracy']['mean']:.3f} ± {summary['Accuracy']['std']:.3f}",
        "ROC_AUC": f"{summary['ROC_AUC']['mean']:.3f} ± {summary['ROC_AUC']['std']:.3f}"
    }]

    import pandas as pd
    df = pd.DataFrame(results)
    print("\n" + "=" * 140)
    print("FINAL CV + BOOTSTRAPPED RESULTS (REAL TRAIN -> SYNTHETIC TEST)")
    print("=" * 140)
    print(df.to_string(index=False))
    df.to_csv("real_train_synthetic_test_cv_bootstrap_results.csv", index=False)
    return df

if __name__ == "__main__":
    # run_ratio_sweep(
    #     train_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/real_train_data_raw.pkl',
    #     test_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/real_test_data.pkl',
    #     # synthetic_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/dataset_sim_Full_Model_20260209_1014_try_20260209_1014_matched.pkl'
    #     synthetic_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/responses_to_review/dataset_sim_Full_Model_Flow_Matching_20260410_2101_matched.pkl'
    #     # synthetic_path="/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/results/dataset_sim_CVAE_matched.pkl"
    #     # synthetic_path="/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/results/dataset_sim_CVAE_matched.pkl"
    # )
    run_ratio_sweep_bootstrapped(
    # # run_ratio_sweep_cv_bootstrapped(
        train_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/real_train_data_raw.pkl',
        test_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/real_test_data.pkl',
        # oldsynthetic_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/dataset_sim_Full_Model_20260209_1014_try_20260209_1014_matched.pkl'
        synthetic_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/responses_to_review/dataset_sim_Full_Model_Flow_Matching_20260410_2101_matched.pkl'
        # synthetic_path="/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/results/dataset_sim_CVAE_matched.pkl"
        # synthetic_path="/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/results/dataset_sim_CVAE_matched.pkl"
    )

    # TARGET_DIR = "/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/responses_to_review"
    # REAL_PATH = "/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/real_test_data.pkl"
    # TRAIN_REAL_PATH = '/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/real_train_data_raw.pkl'
    #
    #
    # def get_latest_file(pattern):
    #     search_path = os.path.join(TARGET_DIR, pattern)
    #     files = glob.glob(search_path)
    #     if not files:
    #         # Fallback for CVAE if it's in a different subfolder
    #         search_path_alt = f"/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/results/{pattern}"
    #         files = glob.glob(search_path_alt)
    #         if not files: return None
    #     return max(files, key=os.path.getctime)

    #     "real": REAL_PATH,
    #     "Flow_Matching_(Ours)": "/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/responses_to_review/dataset_sim_Full_Model_Flow_Matching_20260410_2101_matched.pkl",
    #     "DDIM_(Graph-Aware)": get_latest_file("dataset_sim_Graph_Aware_DDIM_*_matched.pkl"),
    #     "DDPM_(Graph-Aware)": get_latest_file("dataset_sim_Graph_Aware_DDPM_*_matched.pkl"),
    #     "DDIM_(MLP)": get_latest_file("dataset_sim_DiffusionModelMLP_DDIM_*_matched.pkl"),
    #     "DDPM_(MLP)": get_latest_file("dataset_sim_DiffusionModelMLP_DDPM_*_matched.pkl"),
    #     "CVAE_Baseline": get_latest_file("dataset_sim_CVAE_matched.pkl")

    # if __name__ == "__main__":
    # run_real_train_synthetic_test(
    #     train_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/real_train_data_raw.pkl',
    #     # synthetic_path = "/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/real_test_data.pkl"
    #     # synthetic_path='/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/responses_to_review/dataset_sim_Full_Model_Flow_Matching_20260410_2101_matched.pkl'
    #     synthetic_path = "/Users/kb01352/PycharmProjects/whole_pipeline/New/ablations/results/dataset_sim_CVAE_matched.pkl"
    # )
