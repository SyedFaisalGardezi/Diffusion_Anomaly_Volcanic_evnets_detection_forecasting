
import numpy as np
import torch
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg') 
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from labml import experiment, monit
from cfg_ddim import DenoiseDiffusion, gather
from cfg_ddim.experiment_cfg_ddim import Configs
import os
# from metrics import compute_f1_k_auc, compute_roc_k_auc
from sklearn.metrics import roc_auc_score, roc_curve

import torch.nn.functional as F 
from metrics import evaluate, pak_protocol
from sklearn.metrics import roc_auc_score



torch.manual_seed(42)

# Use the same collate function from training.
def collate_fn(batch):
    tensors = [item[0] for item in batch]
    return torch.stack(tensors)

def compute_combined_anomaly_score(original, denoised, relmse_weight=1.0, cosine_weight=1.0):
    """
    Computes a normalized combined anomaly score using:
    - Standard MSE (mean squared error)
    - Cosine Distance

    Args:
        original (torch.Tensor): [C, L]
        denoised (torch.Tensor): [C, L]
        relmse_weight (float): Weight for MSE
        cosine_weight (float): Weight for cosine distance

    Returns:
        score (torch.Tensor): [L] anomaly score per timestep
        mse_score (torch.Tensor): raw MSE score
        cosine_dist (torch.Tensor): raw cosine distance score
    """
    eps = 1e-8

    # ✅ Standard MSE
    mse = (original - denoised) ** 2
    mse_score = mse.sum(dim=0)  # [L]
    
#     # ✅ Relative MSE
#     rel_mse = ((original - denoised) ** 2) / (original ** 2 + eps)
#     rel_mse_score = rel_mse.sum(dim=0)  # [L]
#     mse_score = rel_mse_score
#     
    # ✅ Cosine Distance
    orig_norm = F.normalize(original.T, dim=1)  # [L, C]
    denoised_norm = F.normalize(denoised.T, dim=1)  # [L, C]
    cosine_sim = torch.sum(orig_norm * denoised_norm, dim=1)  # [L]
    cosine_dist = 1. - cosine_sim  # [L]

    # ✅ Normalize both to [0, 1]
    mse_norm = (mse_score - mse_score.min()) / (mse_score.max() - mse_score.min() + eps)
    cosine_norm = (cosine_dist - cosine_dist.min()) / (cosine_dist.max() - cosine_dist.min() + eps)

    # ✅ Weighted combination
    score = relmse_weight * mse_norm + cosine_weight * cosine_norm

    return score, mse_score, cosine_dist



# in evaluate_cgf_ddim.py (or wherever you defined it)
def evaluate_on_split(dataset, sampler, relmse_weight, cosine_weight, device, ensamble):
    all_scores, all_labels = [], []

    for idx in range(len(dataset)):
        sample, times, chunk_label = dataset[idx]
        sample = sample.to(device).unsqueeze(0)  # [1, C, L]

        with torch.no_grad():
            # ensemble denoise
            denoised_runs = [sampler.denoise_sample(sample)
                             for _ in range(ensamble)]
            denoised = torch.stack(denoised_runs, dim=0).mean(dim=0)

            _, mse, _ = compute_combined_anomaly_score(
                original=sample.squeeze(0).to(denoised.device),
                denoised=denoised,
                relmse_weight=relmse_weight,
                cosine_weight=cosine_weight
            )
            all_scores.append(mse.cpu().numpy())

        all_labels.append(np.array(chunk_label, dtype=int))

    return np.concatenate(all_scores), np.concatenate(all_labels) ## i ma returning mse only as scores



# 
# def compute_combined_anomaly_score(original, denoised, relmse_weight=1.0, cosine_weight=1.0):
#     """
#     Computes a combined anomaly score using relative MSE and cosine distance.
# 
#     Args:
#         original (torch.Tensor): [C, L]
#         denoised (torch.Tensor): [C, L]
#         relmse_weight (float): Weight for relative MSE
#         cosine_weight (float): Weight for cosine similarity
# 
#     Returns:
#         score (torch.Tensor): [L] anomaly score per time step
#     """
#     eps = 1e-8
# 
#     # ✅ Relative MSE
#     rel_mse = ((original - denoised) ** 2) / (original ** 2 + eps)
#     rel_mse_score = rel_mse.sum(dim=0)  # [L]
#     
#     # ✅ Standard MSE
#     mse = (original - denoised) ** 2
#     mse_score = mse.sum(dim=0)  # [L]
#     # rel_mse_score = mse_score
#     # ✅ Cosine Similarity
#     orig_norm = F.normalize(original.T, dim=1)  # [L, C]
#     denoised_norm = F.normalize(denoised.T, dim=1)  # [L, C]
#     cosine_sim = torch.sum(orig_norm * denoised_norm, dim=1)  # [L]
#     cosine_dist = 1. - cosine_sim  # [L]
# 
#     # ✅ Combine
#     score = relmse_weight * rel_mse_score + cosine_weight * cosine_dist
#     return score, rel_mse_score, cosine_dist


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TremorDataset(Dataset):
    """
    Evaluation dataset for tremor data with timestamps.
    Returns:
      - chunk:   FloatTensor (5, chunk_size)
      - times:   array of datetime64 for each step
      - labels:  ndarray of 0/1 shape (chunk_size,) marking eruptions
    """

    def __init__(self,
                 file_path: str,
                 start_time: str,
                 end_time: str,
                 chunk_size: int = 30):
        # 1) load and filter by time range
        df = pd.read_csv(file_path, sep=",", parse_dates=['time'])
        df['time'] = pd.to_datetime(df['time'])
        t0 = pd.to_datetime(start_time)
        t1 = pd.to_datetime(end_time)
        df = df.loc[(df['time'] >= t0) & (df['time'] <= t1)].reset_index(drop=True)

        # 2) keep the timestamps and feature arrays
        self.time_series = df['time']  # datetime64[ns]
        feats = ['rsam','mf','hf','dsar','ssam']
        self.data = df[feats].values   # shape (N,5)

        # 3) chunking
        self.chunk_size = chunk_size
        self.num_chunks = len(self.data) // chunk_size
        total = self.num_chunks * chunk_size
        if total != len(self.data):
            self.data = self.data[:total]
            self.time_series = self.time_series.iloc[:total].reset_index(drop=True)

        # 4) eruption instants as numpy datetime64
        self.eruption_times = pd.to_datetime([
            "2012-08-04 16:52:00",
            "2013-08-19 22:23:00",
            "2013-10-03 12:35:00",
            "2016-04-27 09:37:00",
            "2019-12-09 01:11:00"
        ]).values  # numpy.datetime64[]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx: int):
        start = idx * self.chunk_size
        end   = start + self.chunk_size

        # --- chunk data tensor (5, chunk_size)
        chunk_np = self.data[start:end]               # (chunk_size,5)
        chunk    = torch.tensor(chunk_np, dtype=torch.float32).permute(1,0)

        # --- times array (datetime64[ns], length chunk_size)
        times = self.time_series.iloc[start:end].values

        # --- label array: start with all zeros
        labels = np.zeros(len(times), dtype=np.int64)

        # for each eruption, if it falls within this chunk, mark nearest sample
        for et in self.eruption_times:
            # check containment
            if et >= times[0] and et <= times[-1]:
                # compute abs diffs
                diffs = np.abs(times.astype('datetime64[ns]') - et)
                j = diffs.argmin()  # index of nearest sample
                labels[j] = 1

        return chunk, times, labels



"""yahoo dataset """
class YahooSub5Dataset(torch.utils.data.Dataset):
    """
    Univariate/multivariate loader for Yahoo Sub-5, using the CSV
      timestamp,value_0…value_4,ground_truth
    Slices by timestamp range [start_timestamp, end_timestamp], then
    breaks into non-overlapping chunks of length chunk_size.
    Returns (feature_tensor, time_array, label_array) per chunk.
    """

    def __init__(self,
                 file_path: str,
                 chunk_size: int = 30,
                 start_timestamp: int = 1,
                 end_timestamp: int = None):
        # 1) Load whole CSV
        df = pd.read_csv(file_path, sep=",")

        # 2) Determine end_timestamp if not passed
        if end_timestamp is None:
            end_timestamp = int(df['timestamp'].max())

        # 3) Slice by timestamp values
        mask = (df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)
        df = df.loc[mask].reset_index(drop=True)

        # 4) Extract time, features, labels
        self.time_index = df['timestamp'].values.astype(int)          # (N,)
        self.labels     = df['ground_truth'].values.astype(int)      # (N,)
        self.data       = df[['value_0','value_1','value_2','value_3','value_4']]\
                             .values.astype(np.float32)              # (N,5)

        # 5) Chunking setup
        self.chunk_size = chunk_size
        self.num_chunks = len(self.data) // chunk_size

        # truncate to full chunks
        total = self.num_chunks * chunk_size
        self.data       = self.data[:total]
        self.labels     = self.labels[:total]
        self.time_index = self.time_index[:total]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx: int):
        s = idx * self.chunk_size
        e = s + self.chunk_size

        block = self.data[s:e]          # (chunk_size, 5)
        lbl   = self.labels[s:e]        # (chunk_size,)
        times = self.time_index[s:e]    # (chunk_size,)

        # To tensor: (5, chunk_size)
        x = torch.tensor(block, dtype=torch.float32).permute(1, 0)
        y = torch.tensor(lbl,   dtype=torch.int64)

        return x, times, y



## for WaDi Dataset
## laods data and labesl as well and makes a complete label tensor for anamoly and non anamlus.
class WADIDataset(torch.utils.data.Dataset):
    """
    WADI loader that:
     - reads sensor CSV with a 'Datetime' column,
     - reads attack‐interval CSV with columns ['start_dt','end_dt'],
     - builds a per‐timestamp 0/1 label array,
     - chunks both data and labels into (chunk_size)-length segments.
    """
    def __init__(self,
                 data_path: str,
                 label_path: str,
                 start_time,
                 end_time,
                 chunk_size: int = 30):
        # 1) Load sensor data
        df = pd.read_csv(data_path, parse_dates=['Datetime'])
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        # 2) Slice by int or datetime
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
            df = df.iloc[int(start_time):int(end_time)].reset_index(drop=True)
        else:
            st = pd.to_datetime(start_time)
            et = pd.to_datetime(end_time)
            df = df[(df['Datetime'] >= st) & (df['Datetime'] <= et)].reset_index(drop=True)

        # 3) Build label vector from interval file
        lbl_df = pd.read_csv(label_path, parse_dates=['start_dt','end_dt'])
        lbl = np.zeros(len(df), dtype=int)
        times = df['Datetime']
        for _, row in lbl_df.iterrows():
            mask = (times >= row['start_dt']) & (times <= row['end_dt'])
            lbl[mask.values] = 1

        # 4) Drop the timestamp column for model inputs
        self.times = times.values
        data = df.drop(columns=['Datetime']).values
        if data.shape[1] != 123:
            raise ValueError(f"Expected 123 sensor cols, got {data.shape[1]}")

        # 5) Truncate to full chunks
        total = (len(data) // chunk_size) * chunk_size
        self.data = data[:total]
        self.labels = lbl[:total]
        self.times = self.times[:total]
        self.chunk_size = chunk_size
        self.num_chunks = total // chunk_size

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        s = idx * self.chunk_size
        e = s + self.chunk_size
        chunk = self.data[s:e]                           # (chunk_size, 123)
        lab   = self.labels[s:e]                         # (chunk_size,)
        t     = self.times[s:e]                          # (chunk_size,)

        # to tensor: (123, chunk_size)
        x = torch.tensor(chunk, dtype=torch.float32).permute(1,0)
        y = torch.tensor(lab, dtype=torch.int64)         # (chunk_size,)
        return x, t, y


 

class SWATTestDataset(Dataset):
    """
    Dataset for SWAT test data with 51 features and an 'Anomaly' label column.
    Uses integer row indices as pseudo‐time for tracking.
    """

    def __init__(self,
                 file_path: str,
                 chunk_size: int = 50,
                 start_index: int = 0,
                 end_index: int = None):
        # 1) Load CSV
        df = pd.read_csv(file_path)

        # 2) Determine slicing bounds
        if end_index is None:
            end_index = len(df)

        # 3) Extract labels and features
        #    CSV must have a column named "Anomaly"
        self.labels = df["Anomaly"].values[start_index:end_index].astype(np.float32)
        self.data   = df.drop(columns=["Anomaly"]).values[start_index:end_index].astype(np.float32)
        # 4) Build a pseudo‐time index (integers)
        self.time_index = np.arange(start_index, end_index)

        # 5) Chunking parameters
        self.chunk_size  = chunk_size
        self.num_chunks  = len(self.data) // self.chunk_size

        # 6) Truncate to a multiple of chunk_size
        total = self.num_chunks * self.chunk_size
        self.data       = self.data[:total]
        self.labels     = self.labels[:total]
        self.time_index = self.time_index[:total]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index: int):
        # determine start/end for this chunk
        start = index * self.chunk_size
        end   = start + self.chunk_size

        # slice out data, permute to (features, timesteps)
        chunk_data   = self.data[start:end]                            # (chunk_size, 51)
        chunk_tensor = torch.from_numpy(chunk_data).permute(1, 0)      # (51, chunk_size)

        # slice time‐indices and labels
        chunk_time  = self.time_index[start:end]                       # (chunk_size,)
        chunk_label = self.labels[start:end]                           # (chunk_size,)

        return chunk_tensor, chunk_time, chunk_label



class SyntheticDataset(torch.utils.data.Dataset):
    """
    Dataset for synthetic data with 5 features and an 'anomaly' label column.
    Uses row indices as pseudo time for tracking.
    """

    def __init__(self, file_path: str, chunk_size: int = 30, start_index: int = 0, end_index: int = None):
        df = pd.read_csv(file_path, sep=",")

        if end_index is None:
            end_index = len(df)

        # Extract features and labels
        self.labels = df["anomaly"].values[start_index:end_index].astype(np.float32)
        self.data = df.drop(columns=["anomaly"]).values[start_index:end_index].astype(np.float32)
        self.time_index = np.arange(start_index, end_index)

        self.chunk_size = chunk_size
        self.num_chunks = len(self.data) // self.chunk_size

        self.data = self.data[:self.num_chunks * self.chunk_size]
        self.labels = self.labels[:self.num_chunks * self.chunk_size]
        self.time_index = self.time_index[:self.num_chunks * self.chunk_size]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index: int):
        start_idx = index * self.chunk_size
        end_idx = start_idx + self.chunk_size

        chunk_data = self.data[start_idx:end_idx]
        chunk_tensor = torch.tensor(chunk_data).permute(1, 0)
        chunk_time = self.time_index[start_idx:end_idx]
        chunk_label = self.labels[start_idx:end_idx]

        return chunk_tensor, chunk_time, chunk_label




# ✅ Function to Plot Original, Denoised, and Difference
# def plot_results(originals, denoised_list, times_list, img_sav_path, relmse_array, cosine_array, combined_array, title_prefix="Evaluation Results"):
# def plot_results(originals, denoised_list, times_list, img_sav_path, df_combined, df_relmse=None, df_cosine=None, labels_flat=None, title_prefix="Evaluation Results"):
#     
#     """
#     Plots the original, denoised, and difference signals for all test samples.
#     """
#     originals_cat = torch.cat(originals, dim=1)  # [channels, total_time]
#     denoised_cat = torch.cat(denoised_list, dim=1)  # [channels, total_time]
#     diff_cat = originals_cat - denoised_cat
#     times_cat = np.concatenate(times_list)
#     times_cat_trimmed = np.arange(len(times_cat))[3:-3]  # ✅ Use indices
#     # times_num = mdates.date2num(pd.to_datetime(times_cat))
#     # times_cat_trimmed = times_num[3:-3]
# 
#     # labels = ["RSAM", "HF", "MF", "DSAR"]
#     labels = ["F1", "F2", "F3", "F4", "F5"]
#     num_channels = 5  # Only 3 channels used (RSAM, HF, MF)
# 
#     fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
#     axs[0].set_title(f"{title_prefix} - Original Signal")
#     axs[1].set_title(f"{title_prefix} - Denoised Signal")
#     axs[2].set_title(f"{title_prefix} - Difference (Original - Denoised)")
#     axs[3].set_title(f"{title_prefix} - MSE ")
#     
#     # Dictionary to store the first occurrence of threshold crossing across all channels
#     alert_times = {"L1": None, "L2": None, "L3": None, "L4": None}
# 
#     
#     for ch in range(num_channels):
#         l1= 0.1
#         l2=0.2
#         l3=0.3
#         l4=0.35
#         original_values = originals_cat[ch][3:-3].cpu().numpy()
#         denoised_values = denoised_cat[ch][3:-3].cpu().numpy()
#         diff_values = diff_cat[ch][3:-3].cpu().numpy()
#         # diff_values_masked = np.where(np.abs(diff_values) > l1, diff_values, np.nan)
#         diff_values_masked = np.where(np.abs(diff_values) > l1, diff_values, np.nan)
#         # print("times_cat_trimmed shape:", times_cat_trimmed.shape)
# 
#         axs[0].plot(times_cat_trimmed, original_values, '-', label=f"{labels[ch]}")
#         axs[1].plot(times_cat_trimmed, denoised_values, '-', label=f"{labels[ch]}")
#         axs[2].plot(times_cat_trimmed, diff_values_masked, '-', label=f"{labels[ch]}")
#         
#         # # Identify first occurrence of each threshold crossing for the current channel
#         # for i, diff_value in enumerate(diff_values):
#         #     if diff_value > l4:  # Alert L4 (Critical)
#         #         if alert_times["L4"] is None or times_cat_trimmed[i] < alert_times["L4"]:
#         #             alert_times["L4"] = times_cat_trimmed[i]  # Store earliest L4 crossing
#         #     elif diff_value > l3:  # Alert L3
#         #         if alert_times["L3"] is None or times_cat_trimmed[i] < alert_times["L3"]:
#         #             alert_times["L3"] = times_cat_trimmed[i]  # Store earliest L3 crossing
#         #     elif diff_value > l2:  # Alert L2
#         #         if alert_times["L2"] is None or times_cat_trimmed[i] < alert_times["L2"]:
#         #             alert_times["L2"] = times_cat_trimmed[i]  # Store earliest L2 crossing
#         #     elif diff_value > l1:  # Alert L1
#         #         if alert_times["L1"] is None or times_cat_trimmed[i] < alert_times["L1"]:
#         #             alert_times["L1"] = times_cat_trimmed[i]  # Store earliest L1 crossing
#     
#     
#     
# #     # Plot vertical lines at the absolute earliest threshold crossing for each alert level
# #     if alert_times["L4"] is not None:   
# #         axs[2].axhline(y=l4, color='darkred', linestyle=':', linewidth=2, label='Alert L4 (0.4)')
# #         axs[2].axvline(x=alert_times["L4"], color='darkred', linestyle='-', linewidth=3)  # Bold Vertical Line
# #     
# #     if alert_times["L3"] is not None:
# #         axs[2].axhline(y=l3, color='red', linestyle=':', label='Alert L3 (0.3)')
# #         axs[2].axvline(x=alert_times["L3"], color='red', linestyle='--')
# #     
# #     if alert_times["L2"] is not None:
# #         axs[2].axhline(y=l2, color='orange', linestyle=':', label='Alert L2')
# #         axs[2].axvline(x=alert_times["L2"], color='orange', linestyle='--')
# #     
# #     if alert_times["L1"] is not None:
# #         axs[2].axhline(y=l1, color='green', linestyle=':', label='Alert L1')
# #         axs[2].axvline(x=alert_times["L1"], color='green', linestyle='--')
# 
#     for ax in axs[:2]:
#         ax.set_xlabel("Time")
#         ax.set_ylabel("Amplitude")
#         ax.legend(loc='upper left')
# 
#     axs[2].set_xlabel("Time")
#     axs[2].set_ylabel("Amplitude")
#     # axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=45)
#     
#     # ✅ Flatten the arrays
#     df_combined_flat = df_combined.flatten()
#     df_relmse_flat = df_relmse.flatten() if df_relmse is not None else None
#     df_cosine_flat = df_cosine.flatten() if df_cosine is not None else None
# 
#     # ✅ Slice consistently
#     df_combined_trim = df_combined_flat[3:-3]
#     df_relmse_trim = df_relmse_flat[3:-3] if df_relmse is not None else None
#     df_cosine_trim = df_cosine_flat[3:-3] if df_cosine is not None else None
#     
#    
#     
#     # ✅ Plot
#     # axs[3].plot(times_cat_trimmed, df_combined_trim, '-', color='red', label="Combined Score")
#     if df_relmse_trim is not None:
#         axs[3].plot(times_cat_trimmed, df_relmse_trim, '--', color='green', label="RelMSE")
#     if df_cosine_trim is not None:
#         axs[3].plot(times_cat_trimmed, df_cosine_trim, '--', color='orange', label="Cosine Dist")
#     axs[3].axhline(y=0.2, color='gray', linestyle='--', label="Threshold")
#     # axs[3].fill_between(times_cat_trimmed, 0, 1, where=df_combined_trim > 0.25, color='red', alpha=0.2)
# 
#     
#     axs[3].legend(loc="upper right")
#     axs[3].set_ylabel("Score")
#     axs[3].set_xlabel("Time")
# 
#     for ax in axs[:3]:
#         ax.legend(loc='upper right')
#         ax.set_ylabel("Amplitude")
#         
#     # get the current y-axis limits so we can draw bars of height 10
#     y0, y1 = axs[3].get_ylim()
#     bar_top = min(y0 + 10, y1)        # clip at the top of the axis
# 
#     if labels_flat is not None:
#         # only shade where label == 1
#         labels_trim = (labels_flat.astype(int)[3:-3] == 1)
#         axs[3].fill_between(
#             times_cat_trimmed,
#             y0, bar_top,
#             where=labels_trim,
#             color='lightcoral',
#             alpha=0.3,
#             label='True Anomaly'
#         )
# 
#     # re-draw legend to include the new patch
#     axs[3].legend(loc="upper right")
#     axs[3].set_ylim(top=3.0)
# 
#     # Ensure PDF vector format
#     img_sav_path_pdf = img_sav_path.replace(".png", ".pdf")
# 
#     plt.tight_layout()
#     plt.savefig(img_sav_path_pdf, format='pdf', bbox_inches='tight')
#     print(f"✅ Plot saved as high-definition PDF at {img_sav_path_pdf}")
#     plt.show()

### plotting without difference in waveforms

def plot_results(originals, denoised_list, times_list, img_sav_path, df_combined, df_relmse=None, df_cosine=None, labels_flat=None, title_prefix="Evaluation Results"):
    
    """
    Plots the original, denoised, and difference signals for all test samples.
    """
    originals_cat = torch.cat(originals, dim=1)  # [channels, total_time]
    denoised_cat = torch.cat(denoised_list, dim=1)  # [channels, total_time]
    diff_cat = originals_cat - denoised_cat
    
    # concatenate all times, then trim first/last 3 for plotting
    times_cat = np.concatenate(times_list)
    times_trim = times_cat[3:-3]

    # detect whether these are datetimes
    use_dates = False
    try:
        # will succeed if they’re strings or numpy datetime64
        times_trim = pd.to_datetime(times_trim)
        use_dates = True
    except (ValueError, TypeError):
        # leave as numeric indices
        times_trim = times_trim.astype(int)
    times_cat_trimmed = times_trim
    # times_cat = np.concatenate(times_list)
    # times_cat_trimmed = np.arange(len(times_cat))[3:-3]  # ✅ Use indices
    # times_num = mdates.date2num(pd.to_datetime(times_cat))
    # times_cat_trimmed = times_num[3:-3]

    labels = ["RSAM", "HF", "MF", "DSAR"] #
    # labels = ["F1", "F2", "F3", "F4", "F5"]
    num_channels = 4  # Only 3 channels used (RSAM, HF, MF)

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    axs[0].set_title(f"{title_prefix} - Original Signal")
    axs[1].set_title(f"{title_prefix} - Denoised Signal")
    # axs[2].set_title(f"{title_prefix} - Difference (Original - Denoised)")
    axs[2].set_title(f"Anomaly Score ")
    
    # Dictionary to store the first occurrence of threshold crossing across all channels
    alert_times = {"L1": None, "L2": None, "L3": None, "L4": None}
    
    # right before your ch‐loop, compute a good DSAR offset dynamically:
    # take the max absolute value among RSAM, HF and MF
    # other_max = max(
    #     originals_cat[0:3].abs().flatten().max().item(),
    #     denoised_cat[0:3].abs().flatten().max().item()
    # )
    # dsar_offset = other_max * 1.2   # 20% headroom
    
    other_max = originals_cat[0:3].abs().max().item()  # only need originals
    dsar_offset = other_max * 0.2   # just 20% of that max, so it’s small
    
    for ch in range(num_channels):
        l1= 0.1
        l2=0.2
        l3=0.3
        l4=0.35
        original_values = originals_cat[ch][3:-3].cpu().numpy()
        denoised_values = denoised_cat[ch][3:-3].cpu().numpy()
        diff_values = diff_cat[ch][3:-3].cpu().numpy()
        
        # if this is DSAR (the 4th channel), add the offset
        if ch == 3:
            original_values = original_values + dsar_offset
            # denoised_values = denoised_values + dsar_offset
            # diff_values     = diff_values     + dsar_offset
        
        
        # diff_values_masked = np.where(np.abs(diff_values) > l1, diff_values, np.nan)
        # diff_values_masked = np.where(np.abs(diff_values) > l1, diff_values, np.nan)
        # print("times_cat_trimmed shape:", times_cat_trimmed.shape)

        # axs[0].plot(times_cat_trimmed, original_values, '-', label=f"{labels[ch]}")
        # axs[1].plot(times_cat_trimmed, denoised_values, '-', label=f"{labels[ch]}")
        axs[0].plot(times_cat_trimmed, original_values, '-', label=f"{labels[ch]}")
        axs[1].plot(times_cat_trimmed, denoised_values, '-', label=f"{labels[ch]}")
        
        # axs[2].plot(times_cat_trimmed, diff_values_masked, '-', label=f"{labels[ch]}")


    for ax in axs[:2]:
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper left')

    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Amplitude")
    # axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(axs[2].xaxis.get_majorticklabels(), rotation=45)
    
    # ✅ Flatten the arrays
    df_combined_flat = df_combined.flatten()
    df_relmse_flat = df_relmse.flatten() if df_relmse is not None else None
    df_cosine_flat = df_cosine.flatten() if df_cosine is not None else None

    # ✅ Slice consistently
    df_combined_trim = df_combined_flat[3:-3]
    df_relmse_trim = df_relmse_flat[3:-3] if df_relmse is not None else None
    df_cosine_trim = df_cosine_flat[3:-3] if df_cosine is not None else None
    
    threshold_score = 0.15
    
    # ✅ Plot
    # axs[3].plot(times_cat_trimmed, df_combined_trim, '-', color='red', label="Combined Score")
    if df_relmse_trim is not None:
        axs[2].plot(times_cat_trimmed, df_relmse_trim, '--', color='green', label="MSE")
    if df_cosine_trim is not None:
        axs[2].plot(times_cat_trimmed, df_cosine_trim, '--', color='orange', label="Cosine Sim")
    axs[2].axhline(y=threshold_score, color='gray', linestyle='--', label="Threshold")
    # axs[2].fill_between(times_cat_trimmed, 0, 1, where=df_combined_trim > 0.25, color='red', alpha=0.2)

    
    axs[2].legend(loc="upper right")
    axs[2].set_ylabel("Score")
    axs[2].set_xlabel("Time")

    for ax in axs[:2]:
        ax.legend(loc='upper right')
        ax.set_ylabel("Amplitude")
        
    # get the current y-axis limits so we can draw bars of height 10
    y0, y1 = axs[2].get_ylim()
    bar_top = min(y0 + 10, y1)        # clip at the top of the axis

    if labels_flat is not None:
        # only shade where label == 1
        labels_trim = (labels_flat.astype(int)[3:-3] == 1)
        axs[2].fill_between(
            times_cat_trimmed,
            y0, bar_top,
            where=labels_trim,
            color='lightcoral',
            alpha=0.3,
            label='True Anomaly'
        )

    # re-draw legend to include the new patch
    axs[2].legend(loc="upper right")
    # axs[2].set_ylim(top=3.0)

    # Ensure PDF vector format
    img_sav_path_pdf = img_sav_path.replace(".png", ".pdf")

    # plt.tight_layout()
    plt.savefig(img_sav_path_pdf, format='pdf', bbox_inches='tight')
    print(f"✅ Plot saved as high-definition PDF at {img_sav_path_pdf}")
    plt.show()

# Sampler class
class Sampler:
    """
    Sampler for evaluating diffusion models using DDIM and Time-Step Guidance (TSG).
    """
    def __init__(self, diffusion: DenoiseDiffusion, signal_channels: int, seq_length: int, device: torch.device, guidance_scale: float = 3.0, lambda_max: int = 1000):
        self.device = device
        self.seq_length = seq_length
        self.signal_channels = signal_channels
        self.diffusion = diffusion
        self.n_steps = diffusion.n_steps
        self.guidance_scale = guidance_scale
        self.lambda_max = lambda_max  # explicitly save lambda_max here

    # def _sample_x0(self, xt: torch.Tensor, n_steps: int):
    #     """
    #     Runs the reverse DDIM sampling process with Time-Step Guidance.
    #     """
    #     n_samples = xt.shape[0]
    #     for t_inv in monit.iterate('Denoise', n_steps):
    #         t = torch.full((n_samples,), self.n_steps - t_inv - 1, dtype=torch.long, device=self.device)
    #         xt = self.diffusion.ddim_sample(xt, t, guidance_scale=self.guidance_scale)
    #     return xt
    
    def _sample_x0(self, xt: torch.Tensor, lambda_steps: int): ## code for \lambda_max<T i.e 500
        n_samples = xt.shape[0]
        
        for t_inv in monit.iterate('Denoise', lambda_steps):
            # t = torch.full((n_samples,), self.n_steps - t_inv - 1, dtype=torch.long, device=self.device)
            # ✅ fix indexing to correctly handle shortened diffusion (lambda_max)
            t = torch.full((n_samples,), lambda_steps - t_inv - 1, dtype=torch.long, device=self.device)
            # xt = self.diffusion.ddim_sample(xt, t, guidance_scale=self.guidance_scale)
            t_prev = torch.clamp(t - 1, min=0)
            xt = self.diffusion.ddim_sample_heun(xt, t, t_prev)
        return xt

    
    # def denoise_sample(self, sample: torch.Tensor):
    #     """
    #     Adds noise to the input and denoises it using DDIM + TSG.
    #     """
    #     t = torch.full((1,), self.n_steps - 1, dtype=torch.long, device=self.device)
    #     x_noisy = self.diffusion.q_sample(sample, t)
    #     denoised = self._sample_x0(x_noisy, self.n_steps)
    #     return denoised[0].cpu()
    
    def denoise_sample(self, sample: torch.Tensor):
        """
        Adds noise to the input using λ_max and denoises it using DDIM + TSG.
        """
        t = torch.full((1,), self.lambda_max - 1, dtype=torch.long, device=self.device)

        # ✅ Add noise once with fixed epsilon (preserve signal partially)
        eps = torch.randn_like(sample)
        x_noisy = self.diffusion.q_sample_ode(sample, t, eps=eps)

        # ✅ Reverse using shortened steps
        denoised = self._sample_x0(x_noisy, lambda_steps=self.lambda_max)

        return denoised[0].cpu()


# Main function
def main():
    print("Starting evaluation...")
    experiment.evaluate()
    configs = Configs()
    
    
    def get_available_device():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                torch.randn(1).to(f"cuda:{i}")
                return torch.device(f"cuda:{i}")
            except Exception:
                continue
        return torch.device("cpu")

    device = get_available_device()
    print("Using device:", device)
    

    configs.device = device
    

   
    """ eruption dates:
        2012 08 04 16 52 00
        2013 08 19 22 23 00
        2013 10 03 12 35 00
        2016 04 27 09 37 00
        2019 12 09 01 11 00"""
    
    
    d_name =  "Tremor" 
    tremor_type =  "Pavlof"  #     "Whakaari"   #  "Ruapehu" #     "Pavlof"   #          #
    if tremor_type == "Whakaari":
        
        # start_time = "2011-06-01" ## eruption date 2012 08 04 16 52 00 and 2013 08 19
        # end_time = "2014-01-30"
         
        # start_time = "2014-01-01"
        # end_time = "2016-06-01"

        start_time = "2019-01-01"
        end_time = "2020-01-30"
    
    elif tremor_type == "Pavlof":
        start_time = "2015-06-01"  # "2013-06-20", "2014-05-31", "2016-03-27"
        end_time = "2016-04-30"
        
    else:
        start_time = "2007-04-01"  # 2007-09-25 eruption
        end_time = "2007-10-10"
# # #     
    configs.chunk_size = 50 #num_samples #
    ensamble = 3
    WINDOW_SIZE = 4 ## for anomaly score smoothing
    print(f"end_time {end_time}")
    if d_name == "Yahoo":
        print(f"testing the dataset:  {d_name}")
        configs.image_channels = 5  
        test_path = "./cfg/data/processed/yahoo/learningData_yahoo_test_norm.csv"
        configs.dataset = YahooSub5Dataset(test_path, chunk_size=configs.chunk_size,start_timestamp=start_time, end_timestamp=end_time)
        exp = "21"

    elif d_name == "Tremor":
        if tremor_type == "Ruapehu":
            configs.image_channels = 5
            # exp = "22_Tremor"   # # "24_Tremor"  # for Whakaari
            exp = "23_50_Tremor"  #  "23_Tremor"  ## ruapehu
            
            print(f"testing the dataset:  {d_name}")
            test_path = "./cfg/data/raw/Ruapehu_seismic_data_with_ssam_norm_z_percen_m11.csv"
            # test_path = "./cfg/data/raw/Ruapehu_seismic_data_with_ssam_norm_minus1_1.csv"
            # test_path ="./cfg/data/raw/Whakaari_seismic_data_with_ssam_norm_0_1.csv"
            configs.dataset = TremorDataset(test_path, start_time, end_time, chunk_size=configs.chunk_size)
            configs.dataset_val = TremorDataset(test_path, start_time, end_time, chunk_size=configs.chunk_size)
        elif tremor_type == "Whakaari": ##     # 
            configs.image_channels = 5
            # exp = "22_Tremor"   # # "24_Tremor"  # for Whakaari
            exp = "27_Tremor"  #"22_Tremor"  
            
            print(f"testing the dataset:  {d_name}")
            # test_path = "./cfg/data/raw/modified_Whakaari_WIZ_eruption_data_norm_minus1_1_with_SSAM.csv"
            test_path = "./cfg/data/processed/previous_paper_data/WIZ_with_ssam_asof_norm_z_percen_m11.csv"
            "./cfg/data/raw/Whakaari_seismic_data_with_ssam_norm_minus1_1.csv"
            # test_path ="./cfg/data/raw/Whakaari_seismic_data_with_ssam_norm_0_1.csv"
            configs.dataset = TremorDataset(test_path, start_time, end_time, chunk_size=configs.chunk_size)
            configs.dataset_val = TremorDataset(test_path, start_time, end_time, chunk_size=configs.chunk_size)
            
        elif tremor_type == "Pavlof":
            configs.image_channels = 5
            exp = "29_Tremor"  
            
            print(f"testing the dataset:  {d_name}")
            test_path =  "./cfg/data/raw/outside_NZ/PVV_seismic_data_with_ssam_norm_z_percen_m11.csv"
            configs.dataset = TremorDataset(test_path, start_time, end_time, chunk_size=configs.chunk_size)
            configs.dataset_val = TremorDataset(test_path, start_time, end_time, chunk_size=configs.chunk_size)
    



    wtsg=configs.guidance_scale = 3.5  ### ✅ Set guidance scale for CFG
    configs.lambda_max = 500 ## defines from where we should start the backward sampling
    configs.S = 2.0
    configs.T_MIN = 10
    configs.ALPHA = 1.0
    delta = 0.1 ## for AUC computation
    
    configs.data_loader = torch.utils.data.DataLoader(
        configs.dataset, configs.batch_size,
        shuffle=False, pin_memory=True,
        collate_fn=collate_fn, drop_last=True
    )

    configs.init()  # Initialize model and diffusion process
    
    # exp="21"
    event= "Dec-2019"
    img_sav_path = f"./cfg/results/figures/evaluation_results_tsg_exp{exp}_{tremor_type}_{configs.lambda_max}_{start_time}to{end_time}_{d_name}_{wtsg}.png"
    
    configs.save_model_path = f"./cfg/src/cfg_ddim/model_weights/tsg_ddim_TSG_V10_TSG_exp_{exp}.pth"
    
    # configs.save_model_path = "./cfg/src/cfg_ddim/model_weights/tsg_ddim_v9_ode2_heun_pattern_seasonal_TSGconst_1.0_1K_steps.pth"
    
    #"./cfg/src/cfg_ddim/model_weights/tsg_ddim_v9_ode2_heun_pattern_seasonal_TSGconst_1.5.pth"

    
    
    mse_save_dir = f"./cfg/results/mse_analysis/v{exp}_{d_name}_{end_time}_{tremor_type}_{wtsg}"
    os.makedirs(mse_save_dir, exist_ok=True)
    
    configs.load_model()
    """By starting reverse diffusion from x_λ rather than x_T, you preserve partial structure of the original signal, which improves:
    Detection sensitivity
    Preservation of signal features
    Lower false positive rate"""
    

    sampler = Sampler(
        diffusion=configs.diffusion,
        signal_channels=configs.image_channels,
        seq_length=configs.chunk_size,
        device=configs.device,
        guidance_scale=configs.guidance_scale,
        lambda_max=configs.lambda_max
    )

    originals, denoised_list, times_list, labels_list = [], [], [], []
    num_samples = len(configs.dataset)
    print(f"Processing {num_samples} samples...")

    # sampler_val = sampler 
        
    anomaly_threshold = 0.4
    anomaly_indices = []
    # Compute and collect scores
    

    for idx in range(num_samples):
        with torch.no_grad():
            sample, times, chunk_label = configs.dataset[idx]
            sample = sample.to(configs.device).unsqueeze(0)
            # denoised = sampler.denoise_sample(sample)
            
            # ——— ENSEMBLE DENOSING ———
            # run the same sample through the denoiser 5× and take the mean
            denoised_runs = []
            for i in range(ensamble):
                print(f"denoising for iteration {i}")
                denoised_runs.append(sampler.denoise_sample(sample))
            # stack into shape [5, C, L], then average → [C, L]
            denoised = torch.stack(denoised_runs, dim=0).mean(dim=0)
            # ————————————————————————
            
            # ⚙️ Adjust weights as needed
            relmse_weight = 1.0
            cosine_weight = 1.0
            anomaly_threshold = 0.4  # New threshold for combined score

            # 🔁 Compute combined score
            score, rel_mse_per_step, cosine_dist = compute_combined_anomaly_score(
                original=sample.squeeze(0).to(denoised.device),
                denoised=denoised,
                relmse_weight=relmse_weight,
                cosine_weight=cosine_weight
            )

            anomaly_mask = np.abs(score) > anomaly_threshold

            
            if anomaly_mask.any():
                anomaly_indices.append(idx)

            # ✅ Save results
            originals.append(sample.squeeze(0).cpu())
            denoised_list.append(denoised)
            times_list.append(times)
            labels_list.append(chunk_label)    

            # ✅ Plot only if anomaly is detected
            # if anomaly_mask.any():
            time_axis = pd.to_datetime(times)
            plt.figure(figsize=(14, 6), constrained_layout=True)
            for ch in range(sample.shape[1]):
                plt.plot(time_axis, sample[0, ch].cpu(), label=f"Orig - Ch{ch}", alpha=0.7)
                plt.plot(time_axis, denoised[ch], label=f"Denoised - Ch{ch}", linestyle='--')
            for i in range(len(anomaly_mask)):
                if anomaly_mask[i]:
                    plt.axvspan(time_axis[i], time_axis[i], color='red', alpha=0.3)
            plt.title(f"Anomaly Detected at Sample {idx}")
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.legend()
            # plt.tight_layout()
            # plt.savefig(f"./cfg/results/figures/anomaly_plot_sample_{idx}.png")
            plt.close()
                
    labels_flat = np.concatenate(labels_list)    # shape (total_timesteps,)
   
    
    # Compute MSE per sample
    # mse_array = np.array([
    #     F.mse_loss(denoised_list[i], originals[i], reduction='none').mean(dim=0).cpu().numpy()
    #     for i in range(len(originals))
    # ])
    
    combined_array, relmse_array, cosine_array = [], [], []

    for i in range(len(originals)):
        combined, relmse, cosine = compute_combined_anomaly_score(
            originals[i].to(configs.device),
            denoised_list[i].to(configs.device),
            relmse_weight=relmse_weight,
            cosine_weight=cosine_weight
        )
        combined_array.append(combined.detach().cpu().numpy())
        relmse_array.append(relmse.detach().cpu().numpy())
        cosine_array.append(cosine.detach().cpu().numpy())

    # Convert to NumPy arrays
    combined_array = np.array(combined_array)
    relmse_array = np.array(relmse_array)
    cosine_array = np.array(cosine_array)
    
    # flatten everything
    flat_combined = combined_array.flatten()
    flat_mse      = relmse_array.flatten()
    flat_cosine   = cosine_array.flatten()
    # assume times_list is a list of lists of datetime (or epoch) matching each segment
    flat_times    = [t for sub in times_list for t in sub]
    
    # build a single DataFrame
    df_scores = pd.DataFrame({
        "time":     flat_times,
        "combined": flat_combined,
        "mse":      flat_mse,
        "cosine":   flat_cosine
    })

    # if time is an epoch float and you want real datetimes, uncomment:
    # df_scores['time'] = pd.to_datetime(df_scores['time'], unit='s')

    out_csv = os.path.join(mse_save_dir, "anomaly_scores_timeline.csv")
    df_scores.to_csv(out_csv, index=False)
    print(f"Saved combined/mse/cosine scores → {out_csv}")
    
    
    
    
    # flatten to 1D
    test_scores = combined_array.flatten()
    test_labels = gts    = labels_flat.astype(int)
    print(f"  Test: using precomputed {len(test_scores)} time‐steps of scores/labels.")
    
    # Save anomaly score as CSV
    pd.DataFrame(combined_array).to_csv(os.path.join(mse_save_dir, "anomaly_score_combined.csv"), index=False)
    pd.DataFrame(relmse_array).to_csv(os.path.join(mse_save_dir, "anomaly_score_relmse.csv"), index=False)
    pd.DataFrame(cosine_array).to_csv(os.path.join(mse_save_dir, "anomaly_score_cosine.csv"), index=False)


    # Plot combined anomaly score timeline
    flat_combined = combined_array.flatten()
    flat_times = [t for sublist in times_list for t in sublist]

    plt.figure(figsize=(14, 6), constrained_layout=True)
    plt.plot(flat_times, flat_combined, label="Combined Score", color="purple")
    plt.axhline(y=anomaly_threshold, linestyle='--', color='red', label="Threshold")
    plt.title("Combined Anomaly Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.legend()
    # plt.tight_layout()
    plt.savefig(os.path.join(mse_save_dir, "combined_score_over_time.png"))
    plt.close()

    
    print(f"MSE saved to {mse_save_dir}")


    #  Plot results after processing all samples
    # plot_results(originals, denoised_list, times_list, img_sav_path, df_mse ) #
    # print(f"shape of loss arrays")
    plot_results( originals, denoised_list, times_list, img_sav_path,
    df_combined=combined_array, df_relmse=relmse_array, 
    df_cosine=cosine_array, labels_flat = np.concatenate(labels_list)  )


if __name__ == '__main__':
    main()


