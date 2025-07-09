# %%
# from anyio import value
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from Bio import SeqIO
import pingouin as pg
#from inmoose.pycombat import pycombat_norm

import os
import re
import math
from pathlib import Path
HOME = str(Path.home())

# %%
############################################
# General functions
############################################

colors = {
    'blue':    '#377eb8', 
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'red':     '#e41a1c',
    'yellow':  '#dede00'
}


# %%
def IsDefined(x):
    try:
        x
    except NameError:
        return False
    else:
        return True


# %%
def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


# %%
def nip_off_pept(peptide):
    # pept_pattern = "\\.(.+)\\."
    # is equivalent to
    pept_pattern = r"\.(.+)\."
    subpept = re.search(pept_pattern, peptide).group(1)
    return subpept


# %%
def strip_peptide(peptide, nip_off=True):
    if nip_off:
        return re.sub(r"[^A-Za-z]+", '', nip_off_pept(peptide))
    else:
        return re.sub(r"[^A-Za-z]+", '', peptide)


# %%
def get_ptm_pos_in_pept(
    peptide, ptm_label='*', special_chars=r'.]+-=@_!#$%^&*()<>?/\|}{~:['
):
    peptide = nip_off_pept(peptide)
    if ptm_label in special_chars:
        ptm_label = '\\' + ptm_label
    ptm_pos = [m.start() for m in re.finditer(ptm_label, peptide)]
    pos = sorted([val - i - 1 for i, val in enumerate(ptm_pos)])
    return pos


# %%
def get_yst(strip_pept, ptm_aa="YSTyst"):
    return [
        [i, letter.upper()] for i, letter in enumerate(strip_pept) if letter in ptm_aa
    ]


# %%
def get_ptm_info(peptide, residue=None, prot_seq=None, ptm_label='*'):
    if prot_seq != None:
        clean_pept = strip_peptide(peptide)
        pept_pos = prot_seq.find(clean_pept)
        all_yst = get_yst(clean_pept)
        all_ptm = [[pept_pos + yst[0] + 1, yst[1], yst[0]] for yst in all_yst]
        return all_ptm
    if residue != None:
        subpept = nip_off_pept(peptide)
        split_substr = subpept.split(ptm_label)
        res_pos = sorted([int(res) for res in re.findall(r'\d+', residue)])
        first_pos = res_pos[0]
        res_pos.insert(0, first_pos - len(split_substr[0]))
        pept_pos = 0
        all_ptm = []
        for i, res in enumerate(res_pos):
            # print(i)
            if i > 0:
                pept_pos += len(split_substr[i - 1])
            yst_pos = get_yst(split_substr[i])
            if len(yst_pos) > 0:
                for j in yst_pos:
                    ptm = [j[0] + res_pos[i] + 1, j[1], pept_pos + j[0]]
                    all_ptm.append(ptm)
        return all_ptm


# %%
def relable_pept(peptide, label_pos, ptm_label='*'):
    strip_pept = strip_peptide(peptide)
    for i, pos in enumerate(label_pos):
        strip_pept = (
            strip_pept[: (pos + i + 1)] + ptm_label + strip_pept[(pos + i + 1):]
        )
    return peptide[:2] + strip_pept + peptide[-2:]


# %%
def get_phosphositeplus_pos(mod_rsd):
    return [int(re.sub(r"[^0-9]+", '', mod)) for mod in mod_rsd]


# %%
def get_res_names(residues):
    res_names = [
        [res for res in re.findall(r'[A-Z]\d+[a-z\-]+', residue)]
        if residue[0] != 'P'
        else [residue]
        for residue in residues
    ]
    return res_names


# %%
def get_res_pos(residues):
    res_pos = [
        [int(res) for res in re.findall(r'\d+', residue)] if residue[0] != 'P' else [0]
        for residue in residues
    ]
    return res_pos


# %%
def get_sequences_from_fasta(fasta_file):
    prot_seq_obj = SeqIO.parse(fasta_file, "fasta")
    prot_seqs = [seq_item for seq_item in prot_seq_obj]
    return prot_seqs


# %%
def get_protein_res(proteome, uniprot_id, prot_seqs):
    protein = proteome[proteome["uniprot_id"] == uniprot_id]
    protein.reset_index(drop=True, inplace=True)
    prot_seq_search = [seq for seq in prot_seqs if seq.id == uniprot_id]
    prot_seq = prot_seq_search[0]
    sequence = str(prot_seq.seq)
    clean_pepts = [strip_peptide(pept) for pept in protein["peptide"].to_list()]
    protein["clean_pept"] = clean_pepts
    pept_start = [sequence.find(clean_pept) for clean_pept in clean_pepts]
    pept_end = [
        sequence.find(clean_pept) + len(clean_pept) for clean_pept in clean_pepts
    ]
    protein["pept_start"] = pept_start
    protein["pept_end"] = pept_end
    protein["residue"] = [
        [res + str(sequence.find(clean_pept) + i) for i, res in enumerate(clean_pept)]
        for clean_pept in clean_pepts
    ]
    protein_res = protein.explode("residue")
    protein_res.reset_index(drop=True, inplace=True)
    return protein_res


# %%
def adjusted_p_value(pd_series, ignore_na=True, filling_val=1):
    output = pd_series.copy()
    if pd_series.isna().sum() > 0:
        # print("NAs present in pd_series.")
        if ignore_na:
            print("Ignoring NAs.")
            # pd_series =
        else:
            # print("Filling NAs with " + str(filling_val))
            output = sp.stats.false_discovery_control(pd_series.fillna(filling_val))
    else:
        # print("No NAs present in pd_series.")
        output = sp.stats.false_discovery_control(pd_series)
    return output


# %%
############################################
# For both PTM and LiP
############################################


# %%
# This part is to generate the index of dataframes
def generate_index(df, prot_col, level_col=None, id_separator='@', id_col="id"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    if level_col is None:
        df[id_col] = df[prot_col]
    else:
        df[id_col] = df[prot_col] + id_separator + df[level_col]
    df.index = df[id_col].to_list()
    return df


# %%
# def generate_index_pept(pept, prot_col, level_col, id_separator='@'):
#     """_summary_

#     Args:
#         pept (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     pept["id"] = pept[prot_col] + id_separator + pept[level_col]
#     pept.index = pept["id"]
#     return pept


# %%
# def generate_index_prot(prot, uniprot_col):
#     """_summary_

#     Args:
#         prot (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     prot["id"] = prot[uniprot_col]
#     prot.index = prot["id"]
#     return prot


# %%
# this is the function to do the log2 transformation
def log2_transformation(df2transform, int_cols):
    """_summary_

    Args:
        df2transform (_type_): _description_
    """
    df2transform[int_cols] = np.log2(df2transform[int_cols].replace(0, np.nan))
    return df2transform


# %%
# Rolling up the quantification by using log2 medians and plus the log2 of peptide number
def rollup(df_ori, int_cols, rollup_col, id_col="id", id_separator='@', multiply_rollup_counts=True, ignore_NA=True, rollup_func="median"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2**df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
    else:
        if rollup_func.lower() == "median":
            agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
        else:
            ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# %%
# Rolling up the quantification by summing up the raw intensities and then log2 transformation
def rollup_by_sum(df_ori, int_cols, rollup_col, id_col="id", id_separator='@'):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2**df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    agg_methods_2 = {i: "sum" for i in int_cols} 
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# %%
# Rolling up the quantification by using log2 medians and plus the log2 of peptide number
def rollup_by_median(df_ori, int_cols, rollup_col, id_col="id", id_separator='@', multiply_rollup_counts=True, ignore_NA=True):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2**df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    if multiply_rollup_counts:
        if ignore_NA:
            agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
        else:
            agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
    else:
        agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# %%
# Rolling up the quantification by using log2 medians and plus the log2 of peptide number
def rollup_by_mean(df_ori, int_cols, rollup_col, id_col="id", id_separator='@', multiply_rollup_counts=True, ignore_NA=True):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[id_col] = df[id_col] + id_separator + df[rollup_col]

    df[int_cols] = 2**df[int_cols]
    df[int_cols] = df[int_cols].fillna(0)

    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols}
    if multiply_rollup_counts:
        if ignore_NA:
            agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
        else:
            agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
    else:
        agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_1, **agg_methods_2})
    df[int_cols] = np.log2(df[int_cols])
    df[int_cols] = df[int_cols].replace([np.inf, -np.inf], np.nan)
    df.index = df[id_col].to_list()
    return df


# %%
# Here is the function to normalize the data
def median_normalization(df2transform, int_cols, skipna=True, zero_center=False):
    """_summary_

    Args:
        df2transform (_type_): _description_
        int_cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    df_transformed = df2transform.copy()
    if skipna:
        df_filtered = df2transform[df2transform[int_cols].isna().sum(axis=1) == 0].copy()
    else:
        df_filtered = df2transform.copy()

    if zero_center:
        median_correction_T = df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
    else:
        median_correction_T = df_filtered[int_cols].median(axis=0, skipna=True).fillna(0) - df_filtered[int_cols].median(axis=0, skipna=True).fillna(0).mean()
    df_transformed[int_cols] = df_transformed[int_cols].sub(median_correction_T, axis=1)
    return df_transformed


# %%
# Check the missingness by groups
def check_missingness(df, groups, group_cols):
    """_summary_

    Args:
        df (_type_): _description_
    """
    df["Total missingness"] = 0
    for name, cols in zip(groups, group_cols):
        df[f"{name} missingness"] = df[cols].isna().sum(axis=1)
        df["Total missingness"] = df["Total missingness"] + df[f"{name} missingness"]
    return df


# %%
# Here is the function to filter the missingness
def filter_missingness(df, groups, group_cols, missing_thr=0.0):
    """_summary_

    Args:
        df (_type_): _description_
        groups (_type_): _description_
        group_cols (_type_): _description_
        missing_thr (float, optional): _description_. Defaults to 0.0.

    Returns:
        _type_: _description_
    """
    df = check_missingness(df, groups, group_cols)

    df["missing_check"] = 0
    for name, cols in zip(groups, group_cols):
        df["missing_check"] = df["missing_check"] + (df[f"{name} missingness"] > missing_thr * len(cols)).astype(int)
    df_w = df[~(df["missing_check"] > 0)].copy()
    return df_w


# %%
def anova(df, anova_cols, metadata_ori, anova_factors=["Group"]):
    """_summary_

    Args:
        df (_type_): _description_
        anova_cols (_type_): _description_
        metadata_ori (_type_): _description_
        anova_factors (list, optional): _description_. Defaults to ["Group"].

    Returns:
        _type_: _description_
    """
    metadata = metadata_ori[metadata_ori["Sample"].isin(anova_cols)].copy()

    # df = df.drop(columns=["ANOVA_[one-way]_pval", "ANOVA_[one-way]_adj-p"], errors='ignore')

    if len(anova_factors) < 1:
        print("The anova_factors is empty. Please provide the factors for ANOVA analysis. The default factor is 'Group'.")
        anova_factors = ["Group"]
    anova_factor_names = [f"{anova_factors[i]} * {anova_factors[j]}" if i != j else f"{anova_factors[i]}" for i in range(len(anova_factors)) for j in range(i, len(anova_factors))]

    df_w = df[anova_cols].copy()
    # f_stats = []
    f_stats_factors = []
    for row in df_w.iterrows():
        df_id = row[0]
        df_f = row[1]
        df_f = pd.DataFrame(df_f).loc[anova_cols].astype(float)
        df_f = pd.merge(df_f, metadata, left_index=True, right_on="Sample")

        # aov = pg.anova(data=df_f, dv=df_id, between=oneway_factor, detailed=True)
        # if "p-unc" in aov.columns:
        #     p_val = aov[aov["Source"] == oneway_factor]["p-unc"].values[0]
        # else:
        #     p_val = np.nan
        # f_stats.append(pd.DataFrame({"id": [df_id], f"ANOVA_[{oneway_factor}]_pval": [p_val]}))
        try:
            aov_f = pg.anova(data=df_f, dv=df_id, between=anova_factors, detailed=True)
            if "p-unc" in aov_f.columns:
                p_vals = {f"ANOVA_[{anova_factor_name}]_pval": aov_f[aov_f["Source"] == anova_factor_name]["p-unc"].values[0] for anova_factor_name in anova_factor_names}
            else:
                p_vals = {f"ANOVA_[{anova_factor_name}]_pval": np.nan for anova_factor_name in anova_factor_names}
        # except AssertionError as e:
        except Exception as e:
            Warning(f"ANOVA failed for {df_id}: {e}")
            p_vals = {f"ANOVA_[{anova_factor_name}]_pval": np.nan for anova_factor_name in anova_factor_names}
        f_stats_factors.append(pd.DataFrame({"id": [df_id]} | p_vals))

    # f_stats_df = pd.concat(f_stats).reset_index(drop=True)
    # f_stats_df[f"ANOVA_[{oneway_factor}]_adj-p"] = sp.stats.false_discovery_control(f_stats_df[f"ANOVA_[{oneway_factor}]_pval"].fillna(1))
    # f_stats_df.loc[f_stats_df[f"ANOVA_{oneway_factor}_pval"].isna(), f"ANOVA_{oneway_factor}_adj-p"] = np.nan
    # f_stats_df.set_index("id", inplace=True)

    f_stats_factors_df = pd.concat(f_stats_factors).reset_index(drop=True)
    for anova_factor_name in anova_factor_names:
        f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_adj-p"] = sp.stats.false_discovery_control(f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_pval"].fillna(1))
        f_stats_factors_df.loc[f_stats_factors_df[f"ANOVA_[{anova_factor_name}]_pval"].isna(), f"ANOVA_[{anova_factor_name}]_adj-p"] = np.nan
    f_stats_factors_df.set_index("id", inplace=True)
    # f_stats_factors_df.index = f_stats_factors_df["id"].to_list()

    # df = pd.merge(df, f_stats_df, left_index=True, right_index=True)
    df = pd.merge(df, f_stats_factors_df, left_index=True, right_index=True)

    return df


# %%
# Here is the function to do the t-test
# This is same for both protide and protein as well as rolled up protein data. Hopefully this is also the same for PTM data
def pairwise_ttest(df, pairwise_ttest_groups):
    """_summary_

    Args:
        df (_type_): _description_
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        df[pairwise_ttest_group[0]] = (df[pairwise_ttest_group[4]].mean(axis=1) - df[pairwise_ttest_group[3]].mean(axis=1)).fillna(0)
        df[f"{pairwise_ttest_group[0]}_pval"] = sp.stats.ttest_ind(df[pairwise_ttest_group[4]], df[pairwise_ttest_group[3]], axis=1, nan_policy='omit').pvalue
        df[f"{pairwise_ttest_group[0]}_adj-p"] = sp.stats.false_discovery_control(df[f"{pairwise_ttest_group[0]}_pval"].fillna(1))
        df.loc[df[f"{pairwise_ttest_group[0]}_pval"].isna(), f"{pairwise_ttest_group[0]}_adj-p"] = np.nan
    return df


# %%
# calculating the FC and p-values for protein abundances.
def calculate_pairwise_scalars(prot, pairwise_ttest_name=None, sig_type="pval", sig_thr=0.05):
    """_summary_

    Args:
        prot (_type_): _description_
    """
    prot[f"{pairwise_ttest_name}_scalar"] = [prot[pairwise_ttest_name][i] if p < sig_thr else 0 for i, p in enumerate(prot[f"{pairwise_ttest_name}_{sig_type}"])]
    return prot


# %%
def get_prot_abund_scalars(prot, pairwise_ttest_name=None, sig_type="pval", sig_thr=0.05):
    """_summary_

    Args:
        prot (_type_): _description_
        pairwise_ttest_name (_type_, optional): _description_. Defaults to None.
        sig_type (str, optional): _description_. Defaults to "pval".
        sig_thr (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    prot = calculate_pairwise_scalars(prot, pairwise_ttest_name, sig_type, sig_thr)
    scalar_dict = dict(zip(prot.index, prot[f"{pairwise_ttest_name}_scalar"]))
    return scalar_dict


# %%
def calculate_all_pairwise_scalars(prot, pairwise_ttest_groups, sig_type="pval", sig_thr=0.05):
    """_summary_

    Args:
        prot (_type_): _description_
        pairwise_ttest_groups (_type_): _description_
        sig_type (str, optional): _description_. Defaults to "pval".
        sig_thr (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        prot = calculate_pairwise_scalars(prot, pairwise_ttest_group[0], sig_type, sig_thr)
    return prot


# %%
# correct the PTM or LiP data using the protein abundance scalars with significantly changed proteins only
def prot_abund_correction_sig_only(df, prot, pairwise_ttest_groups, uniprot_col, sig_type="pval", sig_thr=0.05):
    """_summary_

    Args:
        df (_type_): _description_
        prot (_type_): _description_
        pairwise_ttest_groups (_type_): _description_
        uniprot_col (_type_): _description_
        sig_type (str, optional): _description_. Defaults to "pval".
        sig_thr (float, optional): _description_. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    for pairwise_ttest_group in pairwise_ttest_groups:
        if pairwise_ttest_group[0] not in prot.columns:
            scalar_dict = get_prot_abund_scalars(prot, pairwise_ttest_group[0], sig_type, sig_thr)
        else:
            scalar_dict = dict(zip(prot.index, prot[f"{pairwise_ttest_group[0]}_scalar"]))
        df[f"{pairwise_ttest_group[0]}_scalar"] = [scalar_dict.get(uniprot_id, 0) for uniprot_id in df[uniprot_col]]
        df[pairwise_ttest_group[4]] = df[pairwise_ttest_group[4]].subtract(df[f"{pairwise_ttest_group[0]}_scalar"], axis=0)
    return df


# %%
# correct the PTM or LiP data using all protein abundance recommended approach
def prot_abund_correction(pept, prot, cols2correct, uniprot_col, non_tt_cols=None):
    """_summary_

    Args:
        pept (_type_): _description_
        prot (_type_): _description_
        cols2correct (_type_): _description_
        uniprot_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    pept_new = []
    if non_tt_cols is None:
        non_tt_cols = cols2correct
    for uniprot_id in pept[uniprot_col].unique():
        pept_sub = pept[pept[uniprot_col] == uniprot_id].copy()
        if uniprot_id in prot[uniprot_col].unique():
            prot_abund_row = prot.loc[uniprot_id, cols2correct]
            prot_abund = prot_abund_row.fillna(0)
            prot_abund_median = prot_abund_row[non_tt_cols].median()
            if prot_abund_median:
                prot_abund_scale = prot_abund_row.div(prot_abund_row).fillna(0) * prot_abund_median
            else:
                prot_abund_scale = prot_abund_row.div(prot_abund_row).fillna(0) * 0
            pept_sub[cols2correct] = pept_sub[cols2correct].sub(prot_abund, axis=1).add(prot_abund_scale, axis=1)
        pept_new.append(pept_sub)
    pept_new = pd.concat(pept_new)

    return pept_new


# Alias the function for PTM data
def prot_abund_correction_TMT(pept, prot, cols2correct, uniprot_col, non_tt_cols=None):
    return prot_abund_correction(pept, prot, cols2correct, uniprot_col, non_tt_cols)


# %%
def count_site_number(df, uniprot_col, site_number_col="site_number"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    site_number = df.groupby(uniprot_col).size()
    site_number.name = site_number_col
    df = pd.merge(df, site_number, left_on=uniprot_col, right_index=True)
    return df


# %%
def count_site_number_with_global_proteomics(df, uniprot_col, id_col, site_number_col="site_number"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    site_number = df.groupby(uniprot_col).size() - 1
    site_number.name = site_number_col
    for uniprot in site_number.index:
        df.loc[df[id_col] == uniprot, site_number_col] = site_number[uniprot]
    return df


# %%
# Batch corrrection using ComBat
#def combat_batch_correction(df, int_cols, metadata, batch_col="Batch", sample_col="Sample", covar_mod=None, par_prior=True, prior_plots=False, mean_only=False, ref_batch=None, precision=None, na_cov_action='raise', **kwargs):
#    df_combat = df.copy()
#    batch_dict = pd.Series(metadata[batch_col].values, index=metadata[sample_col]).to_dict()
#    batch_indices = [batch_dict[int_col] for int_col in int_cols]
#    df_filtered = df_combat[df_combat[int_cols].isna().sum(axis=1) <= 0].copy()
#    df_filtered[int_cols] = pycombat_norm(df_filtered[int_cols], batch_indices, covar_mod, par_prior, prior_plots, mean_only, ref_batch, precision, na_cov_action)
#    return df_filtered


# %%
###########################################
# Specific for PTM
############################################


# %%
# To remove blank columns from TMT tables
def remove_blank_cols(df, blank_cols=None):
    """_summary_

    Args:
        df (_type_): _description_
        blank_cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    if blank_cols is None:
        blank_cols = [col for col in df.columns if 'blank' in col.lower()]
    return df.drop(columns=blank_cols, errors='ignore')


# %%
# To normalize the PTM data by the global protein medians
def PTM_TMT_normalization(df2transform, global_pept, int_cols):
    """_summary_

    Args:
        df2transform (_type_): _description_
    """
    global_filtered = global_pept[global_pept[int_cols].isna().sum(axis=1) == 0].copy()
    global_medians = global_filtered[int_cols].median(axis=0, skipna=True)
    df_transformed = df2transform.copy()
    df_filtered = df2transform[df2transform[int_cols].isna().sum(axis=1) == 0].copy()
    df_medians = df_filtered[int_cols].median(axis=0, skipna=True).fillna(0)
    df_transformed[int_cols] = df_transformed[int_cols].sub(global_medians, axis=1) + df_medians.mean().fillna(0)
    return df_transformed


# %%
# Batch correction for PTM data
def PTM_batch_correction(df4batcor, metadata_ori, batch_correct_samples=None, batch_col="Batch", sample_col="Sample", **kwargs):
    df = df4batcor.copy()
    metadata = metadata_ori.copy()
    if batch_correct_samples is None:
        batch_correct_samples = metadata[sample_col].to_list()
    batch_means = {}
    for batch in metadata[metadata[sample_col].isin(batch_correct_samples)][batch_col].unique():
        df_batch = df[metadata[(metadata[batch_col] == batch) & (metadata[sample_col].isin(batch_correct_samples))][sample_col]].copy()
        df_batch = df_batch[df_batch.isna().sum(axis=1) <= 0].copy()
        df_batch_means = df_batch.mean(axis=0).fillna(0)
        # print(f"Batch {batch} means: {df_batch_means}")
        # print(f"Batch {batch} mean: {df_batch_means.mean()}")
        batch_means.update({batch:  df_batch_means.mean()})
    batch_means = pd.Series(batch_means)
    batch_means_diffs = batch_means - batch_means.mean()
    metadata.index = metadata["Sample"].to_list()
    metadata["batch_correction"] = metadata[batch_col].map(batch_means_diffs)
    df[metadata["Sample"]] = df[metadata["Sample"]].sub(metadata["batch_correction"], axis=1)
    return df


# %%
# Specific for PTM data. This is to roll up the PTM data to the site level
def PTM_rollup_to_site(df_ori, int_cols, uniprot_col, peptide_col, residue_col, residue_sep=';', id_col="id", id_separator='@', site_col="Site", multiply_rollup_counts=True, ignore_NA=True, rollup_func="median"):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0 = {peptide_col: lambda x: '; '.join(x)}
    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
    if multiply_rollup_counts:
        if ignore_NA:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
    else:
        if rollup_func.lower() == "median":
            agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
        elif rollup_func.lower() == "mean":
            agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
        elif rollup_func.lower() == "sum":
            agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
        else:
            ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_0, **agg_methods_1, **agg_methods_2})
    df[site_col] = df[id_col].to_list()
    df.index = df[id_col].to_list()
    return df


# %%
# Specific for PTM data. This is to roll up the PTM data to the site level
def PTM_median_rollup_to_site(df_ori, int_cols, uniprot_col, peptide_col, residue_col, residue_sep=';', id_col="id", id_separator='@', site_col="Site", multiply_rollup_counts=True, ignore_NA=True):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0 = {peptide_col: lambda x: '; '.join(x)}
    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
    if multiply_rollup_counts:
        if ignore_NA:
            agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
        else:
            agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
    else:
        agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_0, **agg_methods_1, **agg_methods_2})
    df[site_col] = df[id_col].to_list()
    df.index = df[id_col].to_list()
    return df


# %%
# Specific for PTM data. This is to roll up the PTM data to the site level
def PTM_mean_rollup_to_site(df_ori, int_cols, uniprot_col, peptide_col, residue_col, residue_sep=';', id_col="id", id_separator='@', site_col="Site", multiply_rollup_counts=True, ignore_NA=True):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = df_ori.reset_index(drop=True)
    info_cols = [col for col in df.columns if col not in int_cols]

    df[residue_col] = df[residue_col].str.split(residue_sep)
    df = df.explode(residue_col)
    df[id_col] = df[uniprot_col] + id_separator + df[residue_col]

    info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
    agg_methods_0 = {peptide_col: lambda x: '; '.join(x)}
    agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
    if multiply_rollup_counts:
        if ignore_NA:
            agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
        else:
            agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
    else:
        agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
    df = df.groupby(id_col, as_index=False).agg({**agg_methods_0, **agg_methods_1, **agg_methods_2})
    df[site_col] = df[id_col].to_list()
    df.index = df[id_col].to_list()
    return df


def combine_multi_PTMs(multi_proteomics, residue_col, uniprot_col, site_col, site_number_col, id_separator='@', id_col="id", type_col="Type", experiment_col="Experiment"):
    proteomics_list = []
    for key, value in multi_proteomics.items():
        if key.lower() == "global":
            prot = value
            prot[type_col] = "Global"
            prot[experiment_col] = "PTM"
            prot[residue_col] = "GLB"
            prot[site_col] = prot[uniprot_col] + id_separator + prot[residue_col]
            proteomics_list.append(prot)
        elif key.lower() == "redox":
            redox = value
            redox[type_col] = "Ox"
            redox[experiment_col] = "PTM"
            redox = count_site_number(redox, uniprot_col, site_number_col)
            proteomics_list.append(redox)
        elif key.lower() == "phospho":
            phospho = value
            phospho[type_col] = "Ph"
            phospho[experiment_col] = "PTM"
            phospho = count_site_number(phospho, uniprot_col, site_number_col)
            proteomics_list.append(phospho)
        elif key.lower() == "acetyl":
            acetyl = value
            acetyl[type_col] = "Ac"
            acetyl[experiment_col] = "PTM"
            acetyl = count_site_number(acetyl, uniprot_col, site_number_col)
            proteomics_list.append(acetyl)
        else:
            KeyError(f"The key {key} is not recognized. Please check the input data.")

    all_ptms = pd.concat(proteomics_list, axis=0, join='outer', ignore_index=True).sort_values(by=[id_col, type_col, experiment_col, site_col]).reset_index(drop=True)
    all_ptms = count_site_number_with_global_proteomics(all_ptms, uniprot_col, id_col, site_number_col)

    return all_ptms


# %%
############################################
# Specific to LiP
############################################


# %%
# This part is filtering all the contaminants and reverse hits
def filter_contaminants_reverse_pept(df, search_tool, ProteinID_col_pept, uniprot_col):
    """_summary_

    Args:
        df (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df = df[(df["Reverse"].isna()) & (df["Potential contaminant"].isna()) & (~df[ProteinID_col_pept].str.contains("(?i)Contaminant")) & (~df[ProteinID_col_pept].str.contains("(?i)REV__")) & (~df[ProteinID_col_pept].str.contains("(?i)CON__"))].copy()
        df[uniprot_col] = df[ProteinID_col_pept]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df = df[(~df[ProteinID_col_pept].str.contains("(?i)contam_"))].copy()
        df[uniprot_col] = df[ProteinID_col_pept]
    else:
        print("The search tool is not specified or not supported yet. The user should provide the tables that have been filtered and without the contaminants and reverse hits.")

    return df


# %%
def filter_contaminants_reverse_prot(df, search_tool, ProteinID_col_prot, uniprot_col):
    """_summary_

    Args:
        df (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df = df[(df["Only identified by site"].isna()) & (df["Reverse"].isna()) & (df["Potential contaminant"].isna()) & (~df[ProteinID_col_prot].str.contains("(?i)Contaminant")) & (~df[ProteinID_col_prot].str.contains("(?i)REV__")) & (~df[ProteinID_col_prot].str.contains("CON__"))].copy()
        df[uniprot_col] = [ids.split(';')[0] for ids in df[ProteinID_col_prot]]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df = df[(~df[ProteinID_col_prot].str.contains("(?i)contam_"))].copy()
        df[uniprot_col] = df[ProteinID_col_prot]
    else:
        print("The search tool is not specified or not supported yet. The user should provide the tables that have been filtered and without the contaminants and reverse hits.")

    return df


# %%
# Filtering out the protein groups with less than 2 peptides
def filtering_protein_based_on_peptide_number(df2filter, PeptCounts_col, search_tool, min_pept_count=2):
    """_summary_

    Args:
        df2filter (_type_): _description_
    """
    if search_tool.lower() == "maxquant":
        df2filter["Pept count"] = [int(count.split(';')[0]) for count in df2filter[PeptCounts_col]]
    elif search_tool.lower() == "msfragger" or search_tool.lower() == "fragpipe":
        df2filter["Pept count"] = df2filter[PeptCounts_col]
    else:
        print("The search tool is not specified or not supported yet. The user should provide the tables that have been filtered and without the contaminants and reverse hits.")
    df2filter = df2filter[df2filter["Pept count"] >= min_pept_count].copy()
    return df2filter


# %%
# This function analyze the trypic pattern of the peptides in pept dataframe
def get_clean_peptides(pept_df, peptide_col, clean_pept_col = "clean_pept"):
    """_summary_

    Args:
        pept_df (_type_): _description_
        peptide_col (_type_): _description_

    Returns:
        _type_: _description_
    """
    clean_pepts = [strip_peptide(pept, nip_off=False) for pept in pept_df[peptide_col].to_list()]
    pept_df[clean_pept_col] = clean_pepts
    return pept_df


# %%
# This function analyze the trypic pattern of the peptides in pept dataframe
def get_tryptic_types(pept_df, prot_seq, peptide_col, clean_pept_col = "clean_pept"):
    seq_len = len(prot_seq)
    # pept_df.reset_index(drop=True, inplace=True)
    if pept_df.shape[0] == 0:
        print("The peptide dataframe is empty. Please check the input dataframe.")
        return
    else:
        if clean_pept_col not in pept_df.columns:
            pept_df = get_clean_peptides(pept_df, peptide_col, clean_pept_col)
        pept_start = [prot_seq.find(clean_pept) + 1 for clean_pept in pept_df[clean_pept_col]]
        pept_end = [prot_seq.find(clean_pept) + len(clean_pept) for clean_pept in pept_df[clean_pept_col]]
        pept_df["pept_start"] = pept_start
        pept_df["pept_end"] = pept_end
        pept_df["pept_type"] = ["Not-matched" if i == 0 else "Tryptic" if ((prot_seq[i-2] in "KR" or i == 1) and (prot_seq[j-1] in "KR" or j == seq_len)) else "Semi-tryptic" if (prot_seq[i-2] in "KR" or prot_seq[j-1] in "KR") else "Non-tryptic" for i,j in zip(pept_start, pept_end)]
    return pept_df



# %%
# This function is to analyze the trypic pattern of the peptides in LiP pept dataframe
def analyze_tryptic_pattern(protein, sequence, pairwise_ttest_groups, groups, description="", peptide_col=None, clean_pept_col="clean_pept", anova_type="[Group]", keep_non_tryptic=True, id_separator="@", sig_type="pval", sig_thr=0.05):
    """_summary_

    Args:
        protein (_type_): _description_
        sequence (_type_): _description_
        description (str, optional): _description_. Defaults to "".
    """
    # protein.reset_index(drop=True, inplace=True)
    seq_len = len(sequence)
    protein["Protein description"] = description
    protein["Protein length"] = seq_len
    protein = get_tryptic_types(protein, sequence, peptide_col, clean_pept_col)
    protein["Tryp Pept num"] = protein[(protein["pept_type"] == "Tryptic")].copy().shape[0]
    protein["Semi Pept num"] = protein[(protein["pept_type"] == "Semi-tryptic")].copy().shape[0]

    ### Abdul made an edit where I set len(groups) >= 2 where originally it was len(groups) > 2
    if len(groups) > 2:
        sig_semi_pepts = protein[(protein["pept_type"] == "Semi-tryptic") & (protein[f"ANOVA_{anova_type}_{sig_type}"] < sig_thr)].copy()
        protein[f"ANOVA_{anova_type} Sig Semi Pept num"] = sig_semi_pepts.shape[0]

    pairwise_ttest_names = [pairwise_ttest_group[0] for pairwise_ttest_group in pairwise_ttest_groups]
    #if sig_semi_pepts.shape[0] != 0:
    #    protein[f"Max absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nanmax(sig_semi_pepts[pairwise_ttest_names].abs().values)
    #    protein[f"Sum absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nansum(sig_semi_pepts[pairwise_ttest_names].abs().values)
    #    protein[f"Median absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nanmedian(sig_semi_pepts[pairwise_ttest_names].abs().values)
    #else:
    #    protein[f"Max absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan
    #    protein[f"Sum absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan
    #    protein[f"Median absFC of All ANOVA_{anova_type} Sig Semi Pept"] = np.nan

    for pairwise_ttest_name in pairwise_ttest_names:
        sig_semi_pepts = protein[(protein["pept_type"] == "Semi-tryptic") & (protein[f"{pairwise_ttest_name}_{sig_type}"] < sig_thr)].copy()
        protein[f"{pairwise_ttest_name} Sig Semi Pept num"] = sig_semi_pepts.shape[0]
        if sig_semi_pepts.shape[0] != 0:
            protein[f"Max absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nanmax(sig_semi_pepts[pairwise_ttest_name].abs().values)
            protein[f"Sum absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nansum(sig_semi_pepts[pairwise_ttest_name].abs().values)
            protein[f"Median absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nanmedian(sig_semi_pepts[pairwise_ttest_name].abs().values)
        else:
            protein[f"Max absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan
            protein[f"Sum absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan
            protein[f"Median absFC of All {pairwise_ttest_name} Sig Semi Pept"] = np.nan

    protein["pept_id"] = [str(protein["pept_start"].to_list()[i]).zfill(4) + '-' + str(protein["pept_end"].to_list()[i]).zfill(4) + id_separator + pept for i, pept in enumerate(protein[peptide_col].to_list())]
    # protein.index = protein["pept_id"]
    not_matched = protein[protein["pept_type"] == "Not-matched"].copy().sort_index()
    not_matched["lytic_group"] = 0
    tryptic = protein[protein["pept_type"] == "Tryptic"].copy().sort_index()
    tryptic["lytic_group"] = 0
    semitryptic = protein[protein["pept_type"] == "Semi-tryptic"].copy().sort_index()
    semitryptic["lytic_group"] = 0
    if keep_non_tryptic:
        nontryptic = protein[protein["pept_type"] == "Non-tryptic"].copy().sort_index()
        nontryptic["lytic_group"] = 0
    for i, idx in enumerate(tryptic.index.to_list()):
        tryptic.loc[idx, "lytic_group"] = i+1
        semitryptic.loc[((semitryptic["pept_start"] == tryptic.loc[idx, "pept_start"]) | (semitryptic["pept_end"] == tryptic.loc[idx, "pept_end"])), "lytic_group"] = i+1
        if keep_non_tryptic:
            nontryptic.loc[((nontryptic["pept_start"].astype(int) > int(tryptic.loc[idx, "pept_start"])) & (nontryptic["pept_start"].astype(int) < int(tryptic.loc[idx, "pept_end"]))), "lytic_group"] = i+1
    if keep_non_tryptic:
        protein_any_tryptic = pd.concat([not_matched, tryptic, semitryptic, nontryptic]).copy()
    else:
        protein_any_tryptic = pd.concat([not_matched, tryptic, semitryptic]).copy()

    return protein_any_tryptic


# %%
# Rollup to site level, NB: this is for individual proteins, because the protein sequence is needed
# This function is to roll up the LiP pept data to the site level with median values
def LiP_rollup_to_site(pept, int_cols, sequence, uniprot_col, residue_col="Residue", uniprot_id="Protein ID (provided by user)", peptide_col="Sequence", clean_pept_col="clean_pept", id_col="id", id_separator="@", pept_type_col="pept_type", site_col="Site", pos_col="Pos", multiply_rollup_counts=True, ignore_NA=True, rollup_func="median"):
    """_summary_

    Args:
        pept (_type_): _description_
        sequence (_type_): _description_
        uniprot_id (str, optional): _description_. Defaults to "".

    Raises:
        ValueError: _description_
    """
    # seq_len = len(sequence)
    if clean_pept_col not in pept.columns.to_list():
        pept = get_tryptic_types(pept, sequence, peptide_col, clean_pept_col)
    if pept.shape[0] > 0:
        pept = get_clean_peptides(pept, peptide_col, clean_pept_col)
        pept[residue_col] = [[res + str(sequence.find(clean_pept)+i+1) for i, res in enumerate(clean_pept)] for clean_pept in pept[clean_pept_col]]
        info_cols = [col for col in pept.columns if col not in int_cols]
        pept = pept.explode(residue_col)
        pept[id_col] = uniprot_id + id_separator + pept[residue_col] + id_separator + pept[pept_type_col]
        # pept[id_col] = uniprot_id + id_separator + pept[residue_col]
        # pept[int_cols] = 2 ** (pept[int_cols])
        # pept_grouped = pept[int_cols].groupby(pept.index).sum(min_count=1)
        # pept_grouped = log2_transformation(pept_grouped)
        # # Lisa Bramer and Kelly Straton suggested to use median of log2 scale values rathen than summing up the intenisty values at linear scale
        info_cols_wo_peptide_col = [col for col in info_cols if col != peptide_col]
        agg_methods_0 = {peptide_col: lambda x: '; '.join(x)}
        agg_methods_1 = {i: lambda x: x.iloc[0] for i in info_cols_wo_peptide_col}
        if multiply_rollup_counts:
            if ignore_NA:
                if rollup_func.lower() == "median":
                    agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.median() for i in int_cols}
                elif rollup_func.lower() == "mean":
                    agg_methods_2 = {i: lambda x: np.log2(len(x)) + x.mean() for i in int_cols}
                elif rollup_func.lower() == "sum":
                    agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
                else:
                    ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
            else:
                if rollup_func.lower() == "median":
                    agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.median() for i in int_cols}
                elif rollup_func.lower() == "mean":
                    agg_methods_2 = {i: lambda x: np.log2(x.notna().sum()) + x.mean() for i in int_cols}
                elif rollup_func.lower() == "sum":
                    agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
                else:
                    ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        else:
            if rollup_func.lower() == "median":
                agg_methods_2 = {i: lambda x: x.median() for i in int_cols}
            elif rollup_func.lower() == "mean":
                agg_methods_2 = {i: lambda x: x.mean() for i in int_cols}
            elif rollup_func.lower() == "sum":
                agg_methods_2 = {i: lambda x: np.log2(np.nansum(2**(x.replace(0, np.nan)))) for i in int_cols}
            else:
                ValueError("The rollup function is not recognized. Please choose from the following: median, mean, sum")
        pept_grouped = pept.groupby(id_col, as_index=False).agg({**agg_methods_0, **agg_methods_1, **agg_methods_2})
        pept_grouped[uniprot_col] = uniprot_id
        pept_grouped[site_col] = [site.split(id_separator)[1] for site in pept_grouped[id_col]]
        pept_grouped[pos_col] = [int(re.sub(r"\D", "", site)) for site in pept_grouped[site_col]]
        pept_grouped.sort_values(by=[pos_col], inplace=True)
        pept_grouped[pept_type_col] = [site.split(id_separator)[-1] for site in pept_grouped[id_col]]
        # pept_grouped.index = uniprot_id + id_separator + pept_grouped["Site"]
        pept_grouped.index = pept_grouped[id_col].to_list()
        return pept_grouped
    else:
        raise ValueError("The pept dataframe is empty. Please check the input dataframe.")
