"""
Statistical testing functions
Compares experimental conditions using appropriate parametric/non-parametric tests
"""

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def run_and_print_statistical_tests(results_dict, ei_ratios, imid_vals, num_repetitions, condition_map):
    """
    Perform statistical tests comparing conditions.

    Checks assumptions (normality, homogeneity of variances), then runs:
    - ANOVA (if assumptions met) + Tukey HSD post-hoc
    - Kruskal-Wallis (if assumptions violated) + Mann-Whitney U post-hoc

    Parameters
    ----------
    results_dict : dict
        Dictionary of metric arrays with shape (len(imid), len(ei_ratios), num_repetitions)
        Keys: "Final Accuracy", "Accuracy at Fixed Samples", "Samples to Threshold", etc.
    ei_ratios : array
        E/I ratio values tested
    imid_vals : array
        Input current values tested
    num_repetitions : int
        Number of repetitions per condition
    condition_map : dict
        Mapping from E/I ratios to condition names (e.g., {0.001: "subcritical"})

    Returns
    -------
    None (prints results to console)
    """
    if num_repetitions < 3:
        print("--- SKIPPING STATISTICAL TESTS: Need at least 3 repetitions per condition for assumption checking. ---")
        return

    print("=" * 80)
    print(" " * 20 + "STATISTICAL ANALYSIS OF RESULTS")
    print("=" * 80)

    for metric_name, data_array in results_dict.items():
        print(f"\n--- Analysis for Metric: {metric_name} ---")

        for i_idx, imid in enumerate(imid_vals):
            print(f"\n--- Condition: Imid = {imid:.4f} nA ---")

            # Collect data for each E/I ratio group
            groups_data = [data_array[i_idx, j_idx, :] for j_idx in range(len(ei_ratios))]

            # Filter out groups with insufficient valid data
            valid_groups = []
            valid_ei_ratios_labels = []
            for k, group in enumerate(groups_data):
                # Keep only groups with at least 3 valid (non-NaN) samples
                if not np.all(np.isnan(group)) and len(group[~np.isnan(group)]) >= 3:
                    valid_groups.append(group[~np.isnan(group)])
                    valid_ei_ratios_labels.append(
                        condition_map.get(ei_ratios[k], f"EI_{ei_ratios[k]:.3f}").capitalize()
                    )

            if len(valid_groups) < 2:
                print("Could not perform test: fewer than two valid E/I ratio groups with sufficient data.")
                continue

            # 1. Check Assumptions for Parametric Tests
            print("\n(1) Checking assumptions for parametric tests (ANOVA)...")
            is_normal = True
            for i, group in enumerate(valid_groups):
                shapiro_stat, shapiro_p = stats.shapiro(group)
                if shapiro_p < 0.05:
                    is_normal = False
                print(f"  - Normality (Shapiro-Wilk) for group '{valid_ei_ratios_labels[i]}': "
                      f"p = {shapiro_p:.4f} ({'Not Normal' if shapiro_p < 0.05 else 'Normal'})")

            levene_stat, levene_p = stats.levene(*valid_groups)
            has_equal_variances = levene_p >= 0.05
            print(f"  - Homogeneity of Variances (Levene's Test): p = {levene_p:.4f} "
                  f"({'Variances are Equal' if has_equal_variances else 'Variances are Unequal'})")

            assumptions_met = is_normal and has_equal_variances
            print(f"\nConclusion: Assumptions for ANOVA are {'MET.' if assumptions_met else 'NOT MET.'}")

            # 2. Perform Appropriate Test
            print("\n(2) Performing appropriate statistical test...")

            if assumptions_met:
                # Parametric: ANOVA
                print("\nUsing parametric test: One-Way ANOVA")
                f_stat, p_val_anova = stats.f_oneway(*valid_groups)
                print(f"  - F-statistic: {f_stat:.4f}")
                print(f"  - P-value: {p_val_anova:.4f}")

                is_significant = p_val_anova < 0.05
                if is_significant:
                    print("  - Result: **Significant difference detected** among E/I ratio groups (p < 0.05).")

                    # Post-hoc: Tukey HSD
                    print("\n  Post-Hoc Test: Tukey's HSD")
                    all_data_flat = np.concatenate(valid_groups)
                    labels_flat = np.concatenate([
                        [label] * len(group) for label, group in zip(valid_ei_ratios_labels, valid_groups)
                    ])
                    tukey_results = pairwise_tukeyhsd(all_data_flat, labels_flat, alpha=0.05)
                    print(tukey_results)
                else:
                    print("  - Result: No significant overall difference detected (p >= 0.05).")

            else:
                # Non-parametric: Kruskal-Wallis
                print("\nUsing non-parametric test: Kruskal-Wallis H-test")
                h_stat, p_val_kw = stats.kruskal(*valid_groups)
                print(f"  - H-statistic: {h_stat:.4f}")
                print(f"  - P-value: {p_val_kw:.4f}")

                is_significant = p_val_kw < 0.05
                if is_significant:
                    print("  - Result: **Significant difference detected** among E/I ratio groups (p < 0.05).")

                    # Post-hoc: Mann-Whitney U with Bonferroni correction
                    print("\n  Post-Hoc Tests: Mann-Whitney U with Bonferroni Correction")
                    pairs = list(combinations(range(len(valid_groups)), 2))
                    corrected_alpha = 0.05 / len(pairs)
                    print(f"  - Bonferroni corrected significance level (alpha): {corrected_alpha:.4f}")

                    for i, j in pairs:
                        u_stat, p_val_mw = stats.mannwhitneyu(
                            valid_groups[i], valid_groups[j], alternative="two-sided"
                        )
                        print(f"  - Comparing '{valid_ei_ratios_labels[i]}' vs '{valid_ei_ratios_labels[j]}': "
                              f"p = {p_val_mw:.4f} "
                              f"{'**SIGNIFICANT**' if p_val_mw < corrected_alpha else ''}")
                else:
                    print("  - Result: No significant overall difference detected (p >= 0.05).")

            print("=" * 80)

    print(" " * 24 + "END OF STATISTICAL ANALYSIS")
    print("=" * 80)


def run_learning_curve_statistics(all_learning_data, imid_param_values, ei_ratio_param_values, condition_map):
    """
    Performs statistical tests (ANOVA + Tukey HSD) at each measurement point 
    of the learning curves to identify when conditions start to diverge.

    Parameters
    ----------
    all_learning_data : array
        Learning curve data with shape (len(imid), len(ei_ratio), num_repetitions)
        Each element is a dict mapping training_size → accuracy
    imid_param_values : array
        Imid values tested
    ei_ratio_param_values : array
        E/I ratio values tested
    condition_map : dict
        Mapping from E/I ratios to condition names

    Returns
    -------
    None (prints results to console)
    """
    print("=" * 80)
    print(" " * 15 + "STATISTICAL ANALYSIS OF LEARNING CURVES")
    print("=" * 80)

    # 1. Flatten learning curves into a DataFrame
    records = []
    for i_idx, imid_val in enumerate(imid_param_values):
        for j_idx, ei_val in enumerate(ei_ratio_param_values):
            condition_name = condition_map.get(ei_val, f"EI_{ei_val:.3f}").capitalize()

            for rep_idx in range(len(all_learning_data[i_idx][j_idx])):
                learning_curve = all_learning_data[i_idx][j_idx][rep_idx]
                if learning_curve and isinstance(learning_curve, dict):
                    for n_samples, accuracy in learning_curve.items():
                        records.append({
                            "Imid": imid_val,
                            "EI_Ratio": ei_val,
                            "Condition": condition_name,
                            "Repetition": rep_idx,
                            "Training_Samples": n_samples,
                            "Accuracy": accuracy
                        })

    if not records:
        print("No learning curve data available for statistical analysis.")
        return

    df = pd.DataFrame(records)

    # 2. Loop through each sample size and perform tests
    sample_sizes = sorted(df["Training_Samples"].unique())

    for size in sample_sizes:
        print(f"\n--- Analysis at Training Size: {size} Samples ---")

        df_size = df[df["Training_Samples"] == size]

        # Group data by condition
        groups_data = [
            df_size[df_size["Condition"] == cond]["Accuracy"].values 
            for cond in df_size["Condition"].unique()
        ]
        labels = list(df_size["Condition"].unique())

        # Filter out conditions with insufficient data
        valid_groups = [g for g in groups_data if len(g) >= 3]
        valid_labels = [labels[i] for i, g in enumerate(groups_data) if len(g) >= 3]

        if len(valid_groups) < 2:
            print("  Cannot perform test: fewer than two conditions with sufficient data.")
            continue

        # Perform ANOVA
        f_stat, p_val_anova = stats.f_oneway(*valid_groups)
        print(f"  One-Way ANOVA result: F-statistic = {f_stat:.4f}, p-value = {p_val_anova:.4f}")

        if p_val_anova < 0.05:
            print("  ANOVA is significant. Performing Tukey's HSD post-hoc test...")

            all_data_flat = np.concatenate(valid_groups)
            labels_flat = np.concatenate([
                [label] * len(group) for label, group in zip(valid_labels, valid_groups)
            ])

            tukey_results = pairwise_tukeyhsd(all_data_flat, labels_flat, alpha=0.05)
            print(tukey_results)
        else:
            print("  No significant overall difference detected among conditions at this sample size.")

    print("=" * 80)
    print(" " * 22 + "END OF LEARNING CURVE ANALYSIS")
    print("=" * 80)
