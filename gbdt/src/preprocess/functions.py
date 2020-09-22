from .utils import pca, drop_low_variace

import pandas as pd


def gene_cell_split_pca_use_row_drop_low_variace(df, g_n_comp, c_n_comp, threshold):
    '''
    split gene and cell
    then, pca gene and cell
    '''
    genes = [col for col in df.columns if col.startswith("g-")]
    cells = [col for col in df.columns if col.startswith("c-")]

    genes_pca = pca(df[genes], g_n_comp, "pca_G_{}")
    cells_pca = pca(df[cells], c_n_comp, "pca_C_{}")
    concat = pd.concat([genes_pca, cells_pca], axis=1)

    df.reset_index(drop=True, inplace=True)
    concat.reset_index(drop=True, inplace=True)
    df = pd.concat([df, concat], axis=1)
    return drop_low_variace(df, threshold)


def gen_cell_concat_pca_drop_variace(df, n_comp, threshold):
    genes_cells = [col for col in df.columns if col.startswith(("g-", "c-"))]
    not_gc = [col for col in df.columns if col not in genes_cells]

    genes_cells_pca = pca(df[genes_cells], n_comp, "pca_{}")
    df.reset_index(drop=True, inplace=True)
    genes_cells_pca.reset_index(drop=True, inplace=True)
    df = pd.concat([df[not_gc], genes_cells_pca], axis=1)
    return drop_low_variace(df, threshold)
