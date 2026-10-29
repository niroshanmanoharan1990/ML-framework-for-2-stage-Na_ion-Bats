"""
Run this ON YOUR MACHINE (where the real model folders exist) to see
exactly what model2_core.py's grouping logic finds in your actual
feature_names.pkl files - no guessing.

Usage:
    python diagnose_model2_features.py
"""
import os
import joblib
import model2_core as core


def main():
    for target_key in core.TARGET_ORDER:
        cfg = core.MODEL_CONFIGS[target_key]
        save_dir = cfg["save_dir"]
        fp = os.path.join(save_dir, core.FEATURE_NAMES_FILENAME)

        print(f"\n{'=' * 70}\n{target_key}  ({save_dir})\n{'=' * 70}")

        if not os.path.exists(fp):
            print(f"  feature_names.pkl NOT FOUND at: {fp}")
            continue

        feature_names = list(joblib.load(fp))
        print(f"  {len(feature_names)} total features")

        grouped = core.build_shap_groups(feature_names)

        additive_hit = grouped.get("Additive concentration", [])
        print(f"\n  Additive concentration group -> {additive_hit if additive_hit else 'NOT FOUND'}")

        print("\n  Full raw feature list (exactly as saved):")
        for f in feature_names:
            print(f"    - {f!r}")

        print("\n  Grouped breakdown (what the GUI will actually build widgets from):")
        for group_key, members in grouped.items():
            tag = " <- ONE-HOT DROPDOWN" if group_key in core.ONE_HOT_DROPDOWN_GROUPS and len(members) > 1 else ""
            print(f"    {group_key:24s}: {members}{tag}")

    print(f"\n{'=' * 70}")
    print("If 'Additive concentration' shows NOT FOUND for every target above,")
    print("none of your 4 trained models were actually fit with a column that")
    print("fuzzy-matches those tokens - the GUI is correctly reflecting that,")
    print("not misbehaving. If you know the real column name (e.g. it's called")
    print("something else in your Excel), tell me the exact name and I'll add")
    print("it to SHAP_GROUP_DEFINITIONS in model2_core.py so it gets picked up.")


if __name__ == "__main__":
    main()
