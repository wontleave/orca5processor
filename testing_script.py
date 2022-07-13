import pandas as pd

if __name__ == "__main__":
    root_ = r"E:\TEST\Orca5Processor_tests\debug\stationary_debug.xlsx"
    req_level_of_theory = "final_gibbs_free_energy -- CPCM(TOLUENE)/wB97X-V/def2-TZVPP//r2scan-3c:XTB"
    folders = []
    df = pd.read_excel(root_, index_col=0)
    sliced = df.loc[req_level_of_theory].iloc[1]  # Duplicates will exist. Idx 0: absolute energy Idx 1: relative
    req_folders = sliced[sliced<0.5].index.values

    # for folder in req_folders:
