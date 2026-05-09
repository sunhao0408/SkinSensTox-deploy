# -*- coding: utf-8 -*-

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem import rdFingerprintGenerator

    MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048
    )

    def mol_to_fp(mol):
        return MORGAN_GENERATOR.GetFingerprint(mol)

except ImportError:
    from rdkit.Chem import AllChem

    def mol_to_fp(mol):
        return AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )


# =========================================================
# 1. Path settings
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "train.xlsx"
TEST_PATH = BASE_DIR / "test.xlsx"
AD_RESULT_PATH = BASE_DIR / "data" / "AD_train_test_result_checked.xlsx"

PROJECT_DIR = BASE_DIR
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pkl"

TOP20_PATH = PROJECT_DIR / "data" / "top20_toxic.csv"

AD_THRESHOLD = 0.263235


# =========================================================
# 2. Page configuration
# =========================================================
st.set_page_config(
    page_title="SkinSensTox",
    page_icon="M",
    layout="wide"
)


# =========================================================
# 3. Core functions
# =========================================================
def standardize_smiles(smiles):
    """
    Standardize SMILES and return canonical SMILES, molecule, fingerprint, and status.
    """
    if pd.isna(smiles):
        return None, None, None, "Empty SMILES"

    smi = str(smiles).strip()

    if smi == "":
        return None, None, None, "Empty SMILES"

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return None, None, None, "Invalid SMILES"

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )

    fp = mol_to_fp(mol)

    return canonical_smiles, mol, fp, "Valid"


def calc_basic_descriptors(mol):
    return {
        "MolWt": round(Descriptors.MolWt(mol), 3),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms()
    }


def get_risk_level(prob):
    if prob is None:
        return "Not available"

    if prob >= 0.80:
        return "High risk"
    elif prob >= 0.50:
        return "Moderate risk"
    else:
        return "Low risk"


@st.cache_resource
def load_train_reference():
    """
    Load training molecules and generate Morgan fingerprints for AD analysis.
    """
    if not TRAIN_PATH.exists():
        return None, None, None, "Training file not found"

    train_df = pd.read_excel(TRAIN_PATH)

    if "SMILES" not in train_df.columns:
        return None, None, None, "The training set does not contain a SMILES column"

    if "Label" not in train_df.columns:
        train_df["Label"] = np.nan

    canonical_list = []
    label_list = []
    fp_list = []

    for _, row in train_df.iterrows():
        canonical_smiles, mol, fp, status = standardize_smiles(row["SMILES"])

        if status == "Valid":
            canonical_list.append(canonical_smiles)
            label_list.append(row["Label"])
            fp_list.append(fp)

    if len(fp_list) == 0:
        return None, None, None, "No valid SMILES found in the training set"

    return canonical_list, label_list, fp_list, "OK"


@st.cache_resource
def load_prediction_model():
    """
    Load the deployed Descriptors-RF model package.
    """
    if MODEL_PATH.exists():
        try:
            model_package = joblib.load(MODEL_PATH)
            return model_package, "OK"
        except Exception as e:
            return None, f"Failed to load model: {e}"

    return None, "Model file not found"


def calculate_ad(fp):
    """
    Assess applicability domain using Morgan fingerprint and Tanimoto similarity.
    """
    train_smiles, train_labels, train_fps, status = load_train_reference()

    if status != "OK":
        return {
            "AD_max_Tanimoto": None,
            "AD_status": "Not available",
            "nearest_train_SMILES": None,
            "nearest_train_Label": None,
            "AD_message": status
        }

    sims = list(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    nearest_idx = int(np.argmax(sims))
    max_sim = float(sims[nearest_idx])

    ad_status = "Inside AD" if max_sim >= AD_THRESHOLD else "Outside AD"

    return {
        "AD_max_Tanimoto": max_sim,
        "AD_status": ad_status,
        "nearest_train_SMILES": train_smiles[nearest_idx],
        "nearest_train_Label": train_labels[nearest_idx],
        "AD_message": "OK"
    }


def calc_descriptors_for_model(mol, model_package):
    """
    Calculate RDKit descriptors using the same preprocessing rules saved during model training.
    """
    desc_names = model_package["desc_names"]
    all_nan_cols = model_package["all_nan_cols"]
    medians = model_package["medians"]
    finite_cols = model_package["finite_cols"]
    huge_cols = model_package["huge_cols"]
    keep_cols = model_package["keep_cols"]

    desc_func_dict = dict(Descriptors._descList)

    values = []

    for name in desc_names:
        func = desc_func_dict.get(name)

        if func is None:
            values.append(np.nan)
            continue

        try:
            v = func(mol)
            if v is None or isinstance(v, str):
                v = np.nan
            else:
                v = float(v)
        except Exception:
            v = np.nan

        values.append(v)

    desc_df = pd.DataFrame([values], columns=desc_names, dtype=np.float64)

    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    desc_df = desc_df.drop(columns=all_nan_cols, errors="ignore")
    desc_df = desc_df.fillna(medians)
    desc_df = desc_df.reindex(columns=finite_cols)
    desc_df = desc_df.drop(columns=huge_cols, errors="ignore")
    desc_df = desc_df.reindex(columns=keep_cols)

    try:
        desc_df = desc_df.fillna(pd.Series(medians).reindex(keep_cols))
    except Exception:
        pass

    desc_df = desc_df.fillna(0)

    X = desc_df.values.astype(np.float64)

    return X


def predict_toxicity(mol):
    """
    Predict skin sensitization using the deployed Descriptors-RF model.
    """
    model_package, status = load_prediction_model()

    if status != "OK":
        return {
            "Pred_probability": None,
            "Pred_label": "Model not loaded",
            "Risk_level": "Not available",
            "Pred_message": status
        }

    try:
        if not isinstance(model_package, dict):
            return {
                "Pred_probability": None,
                "Pred_label": "Unsupported model",
                "Risk_level": "Not available",
                "Pred_message": "best_model.pkl is not a Descriptors-RF model package."
            }

        if model_package.get("model_type") != "Descriptors-RF":
            return {
                "Pred_probability": None,
                "Pred_label": "Unsupported model",
                "Risk_level": "Not available",
                "Pred_message": f"Current model type is {model_package.get('model_type')}, not Descriptors-RF."
            }

        model = model_package["model"]
        X = calc_descriptors_for_model(mol, model_package)

        prob = float(model.predict_proba(X)[0][1])

        threshold = model_package.get("classification_threshold", 0.5)

        pred_label = "Sensitizer" if prob >= threshold else "Non-sensitizer"
        risk_level = get_risk_level(prob)

        return {
            "Pred_probability": prob,
            "Pred_label": pred_label,
            "Risk_level": risk_level,
            "Pred_message": "OK"
        }

    except Exception as e:
        return {
            "Pred_probability": None,
            "Pred_label": "Prediction failed",
            "Risk_level": "Not available",
            "Pred_message": str(e)
        }


def analyze_one_smiles(smiles):
    canonical_smiles, mol, fp, status = standardize_smiles(smiles)

    result = {
        "Input_SMILES": smiles,
        "Canonical_SMILES": canonical_smiles,
        "SMILES_status": status
    }

    if status != "Valid":
        result.update({
            "Pred_probability": None,
            "Pred_label": "Invalid",
            "Risk_level": "Not available",
            "Pred_message": "Invalid SMILES",
            "AD_max_Tanimoto": None,
            "AD_status": "Not available",
            "nearest_train_SMILES": None,
            "nearest_train_Label": None,
            "AD_message": "Invalid SMILES"
        })
        return result, mol

    pred_result = predict_toxicity(mol)
    ad_result = calculate_ad(fp)

    result.update(pred_result)
    result.update(ad_result)

    return result, mol


def display_molecule(mol, title="Molecular structure"):
    if mol is None:
        st.warning("Unable to display molecular structure.")
        return

    img = Draw.MolToImage(mol, size=(420, 300))
    st.image(img, caption=title)


# =========================================================
# 4. Sidebar
# =========================================================
st.sidebar.title("SkinSensTox")
st.sidebar.caption("Skin Sensitization Prediction and Mechanism Visualization")

page = st.sidebar.radio(
    "Select page",
    [
        "Overview",
        "Single Prediction",
        "Batch Prediction",
        "Applicability Domain Analysis",
        "High-Risk Molecules and Mechanistic Interpretation",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.write("Current path settings:")
st.sidebar.code(f"Training set: {TRAIN_PATH}")
st.sidebar.code(f"AD result: {AD_RESULT_PATH}")
st.sidebar.code(f"Model: {MODEL_PATH}")


# =========================================================
# 5. Overview
# =========================================================
if page == "Overview":

    st.title("SkinSensTox")
    st.subheader("Skin Sensitization Toxicity Prediction and Mechanism Visualization Platform")

    st.markdown(
        """
        This platform supports small-molecule skin sensitization risk prediction,
        applicability domain assessment, representative high-risk molecule visualization,
        and mechanistic interpretation.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training molecules", "1082")
    col2.metric("Test molecules", "271")
    col3.metric("AD cutoff", f"{AD_THRESHOLD:.4f}")
    col4.metric("Inside AD ratio", "93.7%")

    st.divider()

    st.markdown("### Workflow")

    st.markdown(
        """
        **SMILES input → Molecular standardization → RDKit descriptor calculation → Descriptors-RF prediction → AD reliability assessment → Structural interpretation → Mechanism visualization**
        """
    )

    st.info(
        "The current version integrates a Descriptors-RF prediction model and retains Morgan-Tanimoto applicability domain assessment."
    )


# =========================================================
# 6. Single Prediction
# =========================================================
elif page == "Single Prediction":

    st.title("Single-Molecule Skin Sensitization Risk Analysis")

    example_smiles = "CC(=O)Oc1ccccc1C(=O)O"

    smiles = st.text_area(
        "Enter a SMILES:",
        value=example_smiles,
        height=100
    )

    run_button = st.button("Run Analysis", type="primary")

    if run_button:

        result, mol = analyze_one_smiles(smiles)

        st.divider()

        left, right = st.columns([1.1, 1.2])

        with left:
            display_molecule(mol)

            st.markdown("### Basic Molecular Properties")

            if mol is not None:
                desc = calc_basic_descriptors(mol)
                st.dataframe(
                    pd.DataFrame([desc]).T.rename(columns={0: "Value"}),
                    use_container_width=True
                )

        with right:
            st.markdown("### Prediction and AD Results")

            st.write(f"**Canonical SMILES:** `{result['Canonical_SMILES']}`")
            st.write(f"**SMILES status:** {result['SMILES_status']}")

            pred_prob = result.get("Pred_probability")

            if pred_prob is not None:
                st.metric("Predicted Skin Sensitization Probability", f"{pred_prob:.4f}")
                st.write(f"**Predicted class:** {result['Pred_label']}")
                st.write(f"**Risk level:** {result['Risk_level']}")

                if result["Pred_label"] == "Sensitizer":
                    st.error("The model predicts that this molecule has skin sensitization risk.")
                else:
                    st.success("The model predicts that this molecule is non-sensitizing.")

            else:
                st.warning(
                    f"Model prediction failed: {result.get('Pred_message', 'Unknown error')}"
                )

            ad_sim = result.get("AD_max_Tanimoto")

            if ad_sim is not None:
                st.metric("AD Max Tanimoto Similarity", f"{ad_sim:.4f}")
                st.write(f"**AD status:** {result['AD_status']}")
                st.write(f"**Nearest training-set molecule:** `{result['nearest_train_SMILES']}`")
                st.write(f"**Nearest training-set molecule label:** {result['nearest_train_Label']}")

                if result["AD_status"] == "Inside AD":
                    st.success(
                        "This molecule is inside the model applicability domain, suggesting relatively higher prediction reliability."
                    )
                else:
                    st.error(
                        "This molecule is outside the model applicability domain; the prediction should be interpreted with caution."
                    )

            else:
                st.warning(
                    f"AD analysis failed: {result.get('AD_message', 'Unknown error')}"
                )


# =========================================================
# 7. Batch Prediction
# =========================================================
elif page == "Batch Prediction":

    st.title("Batch SMILES Analysis")

    st.markdown(
        """
        Upload a CSV or Excel file. The file must contain at least one column named `SMILES`.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Uploaded file preview:")
            st.dataframe(df.head(), use_container_width=True)

            if "SMILES" not in df.columns:
                st.error("The uploaded file does not contain a SMILES column.")

            else:
                if st.button("Run Batch Analysis", type="primary"):

                    results = []

                    for _, row in df.iterrows():
                        result, mol = analyze_one_smiles(row["SMILES"])

                        for col in df.columns:
                            if col not in result and col != "SMILES":
                                result[col] = row[col]

                        results.append(result)

                    result_df = pd.DataFrame(results)

                    st.success("Batch analysis completed.")
                    st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label="Download CSV Results",
                        data=csv_bytes,
                        file_name="SkinSensTox_batch_prediction.csv",
                        mime="text/csv"
                    )

                    output = io.BytesIO()

                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        result_df.to_excel(
                            writer,
                            index=False,
                            sheet_name="prediction_result"
                        )

                    st.download_button(
                        label="Download Excel Results",
                        data=output.getvalue(),
                        file_name="SkinSensTox_batch_prediction.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        except Exception as e:
            st.error(f"File reading or analysis failed: {e}")


# =========================================================
# 8. Applicability Domain Analysis
# =========================================================
elif page == "Applicability Domain Analysis":

    st.title("Applicability Domain Analysis")

    st.markdown(
        """
        This module evaluates whether test-set molecules fall within the training-set chemical space based on Morgan fingerprints and Tanimoto similarity.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AD cutoff", f"{AD_THRESHOLD:.4f}")
    col2.metric("Valid test molecules", "271")
    col3.metric("Inside AD", "254")
    col4.metric("Outside AD", "17")

    st.divider()

    if not AD_RESULT_PATH.exists():
        st.error(f"AD result file not found: {AD_RESULT_PATH}")

    else:
        try:
            summary_df = pd.read_excel(AD_RESULT_PATH, sheet_name="AD_summary")
            test_ad_df = pd.read_excel(AD_RESULT_PATH, sheet_name="test_AD_result")
            train_nn_df = pd.read_excel(AD_RESULT_PATH, sheet_name="train_NN_similarity")

            st.markdown("### AD Summary")
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("### Training-Set Nearest-Neighbor Tanimoto Similarity Distribution")

            if "train_nearest_neighbor_similarity" in train_nn_df.columns:
                fig1 = px.histogram(
                    train_nn_df,
                    x="train_nearest_neighbor_similarity",
                    nbins=40,
                    title="Training-set nearest-neighbor Tanimoto similarity"
                )
                fig1.add_vline(
                    x=AD_THRESHOLD,
                    line_dash="dash",
                    annotation_text=f"AD cutoff = {AD_THRESHOLD:.4f}",
                    annotation_position="top right"
                )
                st.plotly_chart(fig1, use_container_width=True)

            st.markdown("### Test-Set Inside/Outside AD Distribution")

            if "AD_status" in test_ad_df.columns:
                fig2 = px.histogram(
                    test_ad_df,
                    x="AD_status",
                    title="Test-set AD distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Ranked AD_max_Tanimoto Values of Test Compounds")

            if "AD_max_Tanimoto" in test_ad_df.columns:
                temp_df = test_ad_df.copy()
                temp_df = temp_df.sort_values("AD_max_Tanimoto").reset_index(drop=True)
                temp_df["Index"] = np.arange(1, len(temp_df) + 1)

                fig3 = px.scatter(
                    temp_df,
                    x="Index",
                    y="AD_max_Tanimoto",
                    color="AD_status",
                    title="AD_max_Tanimoto values of test compounds"
                )
                fig3.add_hline(
                    y=AD_THRESHOLD,
                    line_dash="dash",
                    annotation_text=f"AD cutoff = {AD_THRESHOLD:.4f}",
                    annotation_position="top left"
                )
                st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to read AD results: {e}")


# =========================================================
# 9. High-Risk Molecules and Mechanistic Interpretation
# =========================================================
elif page == "High-Risk Molecules and Mechanistic Interpretation":

    st.title("High-Risk Molecules and Mechanistic Interpretation")

    st.markdown("### Overall Mechanistic Interpretation")

    st.markdown(
        """
        Overall analysis of predicted targets for representative high-risk skin sensitizers suggests that these molecules may be associated with four major biological processes:
        """
    )

    mechanism_df = pd.DataFrame({
        "Mechanistic category": [
            "Nuclear receptor signaling and hormone/steroid metabolic regulation",
            "Chemical and oxidative stress response",
            "Cell death and damage clearance",
            "Inflammation, immune response, and tissue remodeling"
        ],
        "Representative terms": [
            "nuclear receptors; hormone metabolic process; regulation of steroid metabolic process",
            "cellular response to abiotic stimulus; cellular response to chemical stress; regulation of reactive oxygen species metabolic process",
            "apoptosis; efferocytosis",
            "IL-4/IL-13 signaling; myeloid leukocyte mediated immunity; collagen degradation"
        ],
        "Interpretation": [
            "High-risk molecules may affect nuclear receptor-mediated transcriptional regulation and disturb hormone or steroid metabolic homeostasis.",
            "High-risk molecules may induce chemical stress, oxidative stress, and abnormal ROS metabolism.",
            "High-risk molecules may be involved in cellular damage, apoptosis, and clearance of damaged cells.",
            "High-risk molecules may be associated with inflammatory immune responses and skin tissue remodeling."
        ]
    })

    st.dataframe(mechanism_df, use_container_width=True)

    st.markdown(
        """
        Overall, high-risk skin sensitizers are unlikely to act through a single mechanism.
        Instead, their toxicity may be mediated by multiple processes, including endocrine/nuclear receptor disruption,
        stress-induced damage, altered cell fate, inflammation, and tissue remodeling.
        """
    )

    st.divider()

    st.markdown("### Structure-Based Mechanistic Groups")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Group 1")
        st.markdown("**Long-chain lipophilic reactive**")
        st.write(
            """
            These long-chain lipophilic reactive molecules are mainly associated with lipid/hormone homeostasis disruption,
            abnormal nuclear receptor-related transcriptional regulation, and chemical stress responses.
            """
        )

    with col2:
        st.subheader("Group 2")
        st.markdown("**Sulfur/carbonyl reactive**")
        st.write(
            """
            These sulfur- or carbonyl-containing reactive molecules are mainly associated with MAPK-related stress signaling,
            protein phosphorylation, proteostasis stress, and abnormal drug/lipid metabolism.
            """
        )

    with col3:
        st.subheader("Group 3")
        st.markdown("**Aromatic amine/nitro/halogenated aromatic**")
        st.write(
            """
            These aromatic amine, nitro, or halogenated aromatic molecules are mainly associated with receptor-mediated transcriptional dysregulation,
            abnormal hormone/steroid metabolism, stress-activated kinase cascades, and altered epithelial proliferation or senescence.
            """
        )

    st.divider()

    st.markdown("### Top Toxic Molecules")

    if TOP20_PATH.exists():
        try:
            top20_df = pd.read_csv(TOP20_PATH)

            st.markdown(
                """
                The following table lists representative high-risk skin sensitizers predicted by the model.
                """
            )

            st.dataframe(top20_df, use_container_width=True)

            if "SMILES" not in top20_df.columns:
                st.error("The file top20_toxic.csv does not contain a SMILES column, so molecular structures cannot be rendered.")

            else:
                st.markdown("### Molecular Structure Preview")

                for i, row in top20_df.head(20).iterrows():

                    canonical_smiles, mol, fp, status = standardize_smiles(row["SMILES"])

                    if status != "Valid":
                        with st.expander(f"{i + 1}. Molecule_{i + 1} | Invalid SMILES"):
                            st.warning(f"Invalid SMILES: {row['SMILES']}")
                        continue

                    if "Name" in top20_df.columns:
                        name = row.get("Name")
                    elif "name" in top20_df.columns:
                        name = row.get("name")
                    elif "Compound" in top20_df.columns:
                        name = row.get("Compound")
                    elif "compound" in top20_df.columns:
                        name = row.get("compound")
                    else:
                        name = f"Molecule_{i + 1}"

                    group = None

                    if "Group" in top20_df.columns:
                        group = row.get("Group")
                    elif "group" in top20_df.columns:
                        group = row.get("group")
                    elif "Cluster" in top20_df.columns:
                        group = row.get("Cluster")
                    elif "cluster" in top20_df.columns:
                        group = row.get("cluster")

                    if group is not None:
                        try:
                            if pd.isna(group) or str(group).strip() == "":
                                group = None
                        except Exception:
                            group = None

                    pred_prob = None

                    if "Pred_Prob" in top20_df.columns:
                        pred_prob = row.get("Pred_Prob")
                    elif "Pred_probability" in top20_df.columns:
                        pred_prob = row.get("Pred_probability")
                    elif "pred_prob" in top20_df.columns:
                        pred_prob = row.get("pred_prob")
                    elif "probability" in top20_df.columns:
                        pred_prob = row.get("probability")

                    if pred_prob is None or pd.isna(pred_prob):
                        pred_prob_text = "NA"
                    else:
                        try:
                            pred_prob_text = f"{float(pred_prob):.4f}"
                        except Exception:
                            pred_prob_text = str(pred_prob)

                    if group is None:
                        expander_title = f"{i + 1}. {name} | Pred_Prob = {pred_prob_text}"
                    else:
                        expander_title = f"{i + 1}. {name} | {group} | Pred_Prob = {pred_prob_text}"

                    with st.expander(expander_title):

                        display_molecule(mol)

                        st.write(f"**Canonical SMILES:** `{canonical_smiles}`")
                        st.write(f"**Input SMILES:** `{row['SMILES']}`")

                        if "label" in top20_df.columns:
                            st.write(f"**Original label:** {row.get('label')}")
                        elif "Label" in top20_df.columns:
                            st.write(f"**Original label:** {row.get('Label')}")

                        if "pred_label" in top20_df.columns:
                            st.write(f"**Predicted label:** {row.get('pred_label')}")
                        elif "Pred_Label" in top20_df.columns:
                            st.write(f"**Predicted label:** {row.get('Pred_Label')}")

                        if pred_prob_text != "NA":
                            st.write(f"**Predicted probability:** {pred_prob_text}")

                        if "valid_smiles" in top20_df.columns:
                            st.write(f"**Valid SMILES:** {row.get('valid_smiles')}")
                        elif "Valid_SMILES" in top20_df.columns:
                            st.write(f"**Valid SMILES:** {row.get('Valid_SMILES')}")

                        if "Risk_Level" in top20_df.columns:
                            st.write(f"**Risk level:** {row.get('Risk_Level')}")
                        elif "risk_level" in top20_df.columns:
                            st.write(f"**Risk level:** {row.get('risk_level')}")

                        if "AD_status" in top20_df.columns:
                            st.write(f"**AD status:** {row.get('AD_status')}")
                        elif "ad_status" in top20_df.columns:
                            st.write(f"**AD status:** {row.get('ad_status')}")

                        if "AD_max_Tanimoto" in top20_df.columns:
                            st.write(f"**AD max Tanimoto:** {row.get('AD_max_Tanimoto')}")
                        elif "ad_max_tanimoto" in top20_df.columns:
                            st.write(f"**AD max Tanimoto:** {row.get('ad_max_tanimoto')}")

        except Exception as e:
            st.error(f"Failed to read Top toxic molecules file: {e}")

    else:
        st.info(
            f"Top toxic molecules file was not detected. Please place top20_toxic.csv at: {TOP20_PATH}"
        )


# =========================================================
# 10. About
# =========================================================
elif page == "About":

    st.title("About SkinSensTox")

    st.markdown(
        """
        **SkinSensTox** is a prototype platform for skin sensitization toxicity prediction and mechanism visualization.

        The current version includes:

        1. Single-molecule SMILES analysis;
        2. Batch SMILES analysis;
        3. Descriptors-RF skin sensitization prediction;
        4. Morgan-Tanimoto applicability domain assessment;
        5. Molecular structure visualization;
        6. Mechanistic interpretation of high-risk molecules.
        """
    )

    st.markdown("### Current Model Settings")

    st.code(
        f"""
Prediction model:
    Descriptors-RF

Input features:
    RDKit molecular descriptors

Model file:
    {MODEL_PATH}

Classification threshold:
    0.5
        """
    )

    st.markdown("### Current AD Settings")

    st.code(
        f"""
Morgan fingerprint:
    radius = 2
    nBits = 2048

AD threshold:
    {AD_THRESHOLD:.6f}

Inside AD:
    AD_max_Tanimoto >= {AD_THRESHOLD:.6f}

Outside AD:
    AD_max_Tanimoto < {AD_THRESHOLD:.6f}
        """
    )