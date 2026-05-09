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
# 1. 路径设置
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
# 2. 页面基础设置
# =========================================================
st.set_page_config(
    page_title="SkinSensTox",
    page_icon="🧪",
    layout="wide"
)


# =========================================================
# 3. 基础函数
# =========================================================
def standardize_smiles(smiles):
    """
    SMILES 标准化。
    返回 canonical_smiles, mol, fp, status
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
    读取训练集，生成训练集指纹。
    用于 AD 分析。
    """
    if not TRAIN_PATH.exists():
        return None, None, None, "训练集文件不存在"

    train_df = pd.read_excel(TRAIN_PATH)

    if "SMILES" not in train_df.columns:
        return None, None, None, "训练集缺少 SMILES 列"

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
        return None, None, None, "训练集中没有有效 SMILES"

    return canonical_list, label_list, fp_list, "OK"


@st.cache_resource
def load_prediction_model():
    """
    读取网站用 Descriptors-RF 模型包。
    best_model.pkl 应该位于：
    D:\shsh\SkinSensTox_streamlit\models\best_model.pkl
    """
    if MODEL_PATH.exists():
        try:
            model_package = joblib.load(MODEL_PATH)
            return model_package, "OK"
        except Exception as e:
            return None, f"模型读取失败：{e}"

    return None, "模型文件不存在"


def calculate_ad(fp):
    """
    基于 Morgan fingerprint + Tanimoto similarity 进行 AD 判断。
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
    按训练 Descriptors-RF 时保存的规则计算 RDKit descriptors。
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
    使用网站导出的 Descriptors-RF 模型进行皮肤致敏预测。
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
                "Pred_message": "best_model.pkl 不是 Descriptors-RF 模型包。"
            }

        if model_package.get("model_type") != "Descriptors-RF":
            return {
                "Pred_probability": None,
                "Pred_label": "Unsupported model",
                "Risk_level": "Not available",
                "Pred_message": f"当前模型类型为 {model_package.get('model_type')}，不是 Descriptors-RF。"
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
        st.warning("无法显示分子结构。")
        return

    img = Draw.MolToImage(mol, size=(420, 300))
    st.image(img, caption=title)


# =========================================================
# 4. 侧边栏
# =========================================================
st.sidebar.title("SkinSensTox")
st.sidebar.caption("Skin sensitization prediction and mechanism visualization")

page = st.sidebar.radio(
    "选择页面",
    [
        "首页 Overview",
        "单分子预测 Single Prediction",
        "批量预测 Batch Prediction",
        "AD 适用域分析",
        "高风险分子与机制解释",
        "关于 About"
    ]
)

st.sidebar.divider()
st.sidebar.write("当前路径设置：")
st.sidebar.code(f"训练集: {TRAIN_PATH}")
st.sidebar.code(f"AD结果: {AD_RESULT_PATH}")
st.sidebar.code(f"模型: {MODEL_PATH}")


# =========================================================
# 5. 首页
# =========================================================
if page == "首页 Overview":

    st.title("SkinSensTox")
    st.subheader("皮肤致敏毒性预测与机制可视化平台")

    st.markdown(
        """
        本平台用于小分子皮肤致敏风险预测、适用域可靠性判断、
        高风险分子结构分类以及潜在机制解释。
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("训练集分子数", "1082")
    col2.metric("测试集分子数", "271")
    col3.metric("AD 阈值", f"{AD_THRESHOLD:.4f}")
    col4.metric("Inside AD 比例", "93.7%")

    st.divider()

    st.markdown("### 研究流程")

    st.markdown(
        """
        **SMILES 输入 → 分子标准化 → RDKit descriptors 计算 → Descriptors-RF 模型预测 → AD 可靠性判断 → 结构解释 → 机制可视化**
        """
    )

    st.info(
        "当前版本已接入 Descriptors-RF 预测模型，并保留 Morgan-Tanimoto AD 适用域判断。"
    )


# =========================================================
# 6. 单分子预测
# =========================================================
elif page == "单分子预测 Single Prediction":

    st.title("单分子皮肤致敏风险分析")

    example_smiles = "CC(=O)Oc1ccccc1C(=O)O"

    smiles = st.text_area(
        "输入一个 SMILES：",
        value=example_smiles,
        height=100
    )

    run_button = st.button("开始分析", type="primary")

    if run_button:

        result, mol = analyze_one_smiles(smiles)

        st.divider()

        left, right = st.columns([1.1, 1.2])

        with left:
            display_molecule(mol)

            st.markdown("### 基础分子性质")

            if mol is not None:
                desc = calc_basic_descriptors(mol)
                st.dataframe(
                    pd.DataFrame([desc]).T.rename(columns={0: "Value"}),
                    use_container_width=True
                )

        with right:
            st.markdown("### 预测与 AD 结果")

            st.write(f"**Canonical SMILES:** `{result['Canonical_SMILES']}`")
            st.write(f"**SMILES 状态:** {result['SMILES_status']}")

            pred_prob = result.get("Pred_probability")

            if pred_prob is not None:
                st.metric("皮肤致敏预测概率", f"{pred_prob:.4f}")
                st.write(f"**预测类别:** {result['Pred_label']}")
                st.write(f"**风险等级:** {result['Risk_level']}")

                if result["Pred_label"] == "Sensitizer":
                    st.error("模型预测该分子具有皮肤致敏风险。")
                else:
                    st.success("模型预测该分子为非皮肤致敏分子。")

            else:
                st.warning(
                    f"模型预测未完成：{result.get('Pred_message', 'Unknown error')}"
                )

            ad_sim = result.get("AD_max_Tanimoto")

            if ad_sim is not None:
                st.metric("AD 最大 Tanimoto 相似度", f"{ad_sim:.4f}")
                st.write(f"**AD 状态:** {result['AD_status']}")
                st.write(f"**最相似训练集分子:** `{result['nearest_train_SMILES']}`")
                st.write(f"**最相似训练集分子 Label:** {result['nearest_train_Label']}")

                if result["AD_status"] == "Inside AD":
                    st.success("该分子位于模型适用域内，预测可靠性相对较高。")
                else:
                    st.error("该分子位于模型适用域外，预测结果应谨慎解释。")

            else:
                st.warning(
                    f"AD 分析未完成：{result.get('AD_message', 'Unknown error')}"
                )


# =========================================================
# 7. 批量预测
# =========================================================
elif page == "批量预测 Batch Prediction":

    st.title("批量 SMILES 分析")

    st.markdown(
        """
        上传 CSV 或 Excel 文件，文件中至少需要包含一列 `SMILES`。
        """
    )

    uploaded_file = st.file_uploader(
        "上传文件",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("上传文件预览：")
            st.dataframe(df.head(), use_container_width=True)

            if "SMILES" not in df.columns:
                st.error("文件中缺少 SMILES 列。")

            else:
                if st.button("开始批量分析", type="primary"):

                    results = []

                    for _, row in df.iterrows():
                        result, mol = analyze_one_smiles(row["SMILES"])

                        for col in df.columns:
                            if col not in result and col != "SMILES":
                                result[col] = row[col]

                        results.append(result)

                    result_df = pd.DataFrame(results)

                    st.success("批量分析完成。")
                    st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label="下载 CSV 结果",
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
                        label="下载 Excel 结果",
                        data=output.getvalue(),
                        file_name="SkinSensTox_batch_prediction.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        except Exception as e:
            st.error(f"文件读取或分析失败：{e}")


# =========================================================
# 8. AD 适用域分析
# =========================================================
elif page == "AD 适用域分析":

    st.title("AD 适用域分析")

    st.markdown(
        """
        本模块基于 Morgan fingerprint 与 Tanimoto similarity 评估测试集分子是否位于训练集化学空间内。
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AD 阈值", f"{AD_THRESHOLD:.4f}")
    col2.metric("测试集有效分子数", "271")
    col3.metric("Inside AD", "254")
    col4.metric("Outside AD", "17")

    st.divider()

    if not AD_RESULT_PATH.exists():
        st.error(f"未找到 AD 结果文件：{AD_RESULT_PATH}")

    else:
        try:
            summary_df = pd.read_excel(AD_RESULT_PATH, sheet_name="AD_summary")
            test_ad_df = pd.read_excel(AD_RESULT_PATH, sheet_name="test_AD_result")
            train_nn_df = pd.read_excel(AD_RESULT_PATH, sheet_name="train_NN_similarity")

            st.markdown("### AD Summary")
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("### 训练集最近邻 Tanimoto 相似度分布")

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

            st.markdown("### 测试集 Inside / Outside AD 分布")

            if "AD_status" in test_ad_df.columns:
                fig2 = px.histogram(
                    test_ad_df,
                    x="AD_status",
                    title="Test-set AD distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 测试集 AD_max_Tanimoto 排序图")

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
            st.error(f"读取 AD 结果失败：{e}")


# =========================================================
# 9. 高风险分子与机制解释
# =========================================================
elif page == "高风险分子与机制解释":

    st.title("高风险分子与机制解释")

    st.markdown("### 总体机制解释")

    st.markdown(
        """
        对核心高风险皮肤毒性分子的预测靶点进行整体分析后发现，
        这些分子整体上可能涉及四类主要生物学过程：
        """
    )

    mechanism_df = pd.DataFrame({
        "机制类别": [
            "核受体信号及激素/甾体代谢调控",
            "化学与氧化应激反应",
            "细胞死亡与损伤清除",
            "炎症免疫及组织重塑"
        ],
        "代表条目": [
            "nuclear receptors; hormone metabolic process; regulation of steroid metabolic process",
            "cellular response to abiotic stimulus; cellular response to chemical stress; regulation of reactive oxygen species metabolic process",
            "apoptosis; efferocytosis",
            "IL-4/IL-13 signaling; myeloid leukocyte mediated immunity; collagen degradation"
        ],
        "机制解释": [
            "提示高风险分子可能影响核受体相关转录调控，并干扰激素或甾体代谢稳态。",
            "提示高风险分子可能诱导化学应激、氧化应激和 ROS 代谢异常。",
            "提示高风险分子可能参与细胞损伤、凋亡以及损伤细胞清除过程。",
            "提示高风险分子可能与炎症免疫反应和皮肤组织重塑过程相关。"
        ]
    })

    st.dataframe(mechanism_df, use_container_width=True)

    st.markdown(
        """
        总体来看，高风险皮肤毒性分子可能并非通过单一机制发挥作用，
        而更可能通过内分泌/核受体扰动、应激损伤、细胞命运改变以及炎症和组织重塑等多种途径共同介导毒性。
        """
    )

    st.divider()

    st.markdown("### 结构分组机制")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Group 1")
        st.markdown("**Long-chain lipophilic reactive**")
        st.write(
            """
            这类长链高疏水反应性分子主要偏向脂质/激素稳态扰动、
            核受体相关转录调控异常以及化学应激反应。
            """
        )

    with col2:
        st.subheader("Group 2")
        st.markdown("**Sulfur/carbonyl reactive**")
        st.write(
            """
            这类含硫/含羰基反应性分子主要偏向 MAPK 相关应激信号、
            蛋白磷酸化、蛋白稳态压力以及药物代谢/脂质代谢异常。
            """
        )

    with col3:
        st.subheader("Group 3")
        st.markdown("**Aromatic amine/nitro/halogenated aromatic**")
        st.write(
            """
            这类芳香胺/硝基/卤代芳香分子主要偏向受体介导的转录调控失衡、
            激素/甾体代谢异常、应激激酶级联激活以及上皮细胞增殖和衰老状态改变。
            """
        )

    st.divider()

    st.markdown("### Top toxic molecules")

    if TOP20_PATH.exists():
        try:
            top20_df = pd.read_csv(TOP20_PATH)

            st.markdown(
                """
                下表展示模型预测得到的高风险皮肤致敏分子。
                """
            )

            st.dataframe(top20_df, use_container_width=True)

            if "SMILES" not in top20_df.columns:
                st.error("top20_toxic.csv 中缺少 SMILES 列，无法绘制分子结构。")

            else:
                st.markdown("### 分子结构预览")

                for i, row in top20_df.head(20).iterrows():

                    canonical_smiles, mol, fp, status = standardize_smiles(row["SMILES"])

                    if status != "Valid":
                        with st.expander(f"{i + 1}. Molecule_{i + 1} | Invalid SMILES"):
                            st.warning(f"该分子 SMILES 无效：{row['SMILES']}")
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
            st.error(f"读取 Top20 文件失败：{e}")

    else:
        st.info(
            f"当前未检测到 Top20 文件。后续可把 top20_toxic.csv 放到：{TOP20_PATH}"
        )


# =========================================================
# 10. About
# =========================================================
elif page == "关于 About":

    st.title("About SkinSensTox")

    st.markdown(
        """
        **SkinSensTox** 是一个用于皮肤致敏毒性预测和机制可视化的原型平台。

        当前版本包含：

        1. 单分子 SMILES 分析；
        2. 批量 SMILES 分析；
        3. Descriptors-RF 皮肤致敏预测；
        4. Morgan-Tanimoto AD 适用域判断；
        5. 分子结构图展示；
        6. 高风险分子机制解释。
        """
    )

    st.markdown("### 当前模型设置")

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

    st.markdown("### 当前 AD 设置")

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