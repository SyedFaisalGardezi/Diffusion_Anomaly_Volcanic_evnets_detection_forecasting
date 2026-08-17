"""

"""
import sys


from typing import List
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.utils.data
import torchvision
from PIL import Image
import os

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs

from cfg_ddim.unet import UNet
from cfg_ddim import DenoiseDiffusion  # Import updated diffusion model



def collate_fn(batch):
    # Stack the list of tensors into a batch of shape (B, 3, 300)
    return torch.stack(batch)

def get_available_device():
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            torch.randn(1).to(f"cuda:{i}")
            return torch.device(f"cuda:{i}")
        except Exception:
            continue
    return torch.device("cpu")

class Configs(BaseConfigs):
    """
    ## Configurations
    """

    # Dynamically select available CUDA device
    device: torch.device = get_available_device()
    # device: torch.device = DeviceConfigs() #"cpu" #
 
    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Number of channels in the image. $3$ for RGB.
    image_channels: int = 3
    # Image size
    chunk_size : int = 300
    # image_size : int =32
    # Number of channels in the initial feature map
    n_channels: int = 64
    # The list of channel numbers at each resolution.
    # The number of channels is `channel_multipliers[i] * n_channels`
    channel_multipliers: List[int] = [1, 2, 2, 4]
    # The list of booleans that indicate whether to use attention at each resolution
    is_attention: List[int] = [False, False, False, True]
    
    # Number of residual blocks per resolution level
    n_blocks = 2 
    
    # Number of time steps $T$
    n_steps: int = 1_000
    # Number of time steps $\lambda$, Set λ to 250 or 500 for inference so that we sample from x_t where we have some information of orignal signal as well rather the complete noise at T.
    lambda_max: int = 5_00
    
    # Batch size
    batch_size: int = 64
    # Number of samples to generate
    n_samples: int = 16
    # Learning rate
    learning_rate: float = 2e-5

    # Number of training epochs
    epochs: int = 1000
    
    # dimensions of data i.e. rsam,hf,mf, dsar
    dim_data: int = 4

    # Dataset
    dataset: torch.utils.data.Dataset
    dataset_val:  torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader

    # Adam optimizer
    optimizer: torch.optim.Adam
    
    # Define model save path
    save_model_path: str = "path_in_training_code.pth"
    
    #define the data path
    dataset_path: str = "path_to be given in training code."
    data_name : str = 'Tremor'  ## 'WADI'
    
    guidance_scale: float = 1.0
    alpha : float = 1.0
    T_MIN : int = 50
    T_MAX :int = lambda_max-T_MIN
    ALPHA: float = 1.0  ## its the power for TSG noise controling. t_emb = t_emb + s(t^α)n where n ∼ N (000,III )
    S: float = 1.0
    exp: str = "exp_10"
    
    def init(self):
        """
        Initialize the diffusion model with Classifier-Free Guidance and DDIM Sampling
        """
        self.eps_model = UNet(image_channels=self.image_channels, 
                              n_channels=self.n_channels, 
                              ch_mults=self.channel_multipliers, 
                              is_attn=self.is_attention, 
                              n_blocks=self.n_blocks).to(self.device)
        ### based on CFG
        # self.diffusion = DenoiseDiffusion(
        #     eps_model=self.eps_model,
        #     n_steps=self.n_steps,
        #     device=self.device,
        #     guidance_scale=3.0  # Enable Classifier-Free Guidance
        # )
        
        ## TSG based diffusion
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=self.n_steps,
            lambda_max=self.lambda_max,
            device=self.device,
            guidance_scale=self.guidance_scale,  # Still used as w_TSG
            T_MIN=self.T_MIN,                          # ✅ Apply TSG only from t=50
            T_MAX=self.T_MAX,                         # ✅ to t=950
            S=self.S,                    # ✅ Noise strength coefficient
            ALPHA= self.ALPHA                        # ✅ Power exponent
        )
        
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size,
                                                       shuffle=True, pin_memory=True, collate_fn=collate_fn, drop_last=True)
        self.optimizer = torch.optim.Adam(self.eps_model.parameters(), lr=self.learning_rate)
        # tracker.set_image("sample", True)

    def sample(self):
        """
        ### Sample images using Heun's Method (2nd-order DDIM Sampling)
        """
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.image_channels, self.chunk_size], device=self.device)

            # Prepare full list of time steps
            time_steps = list(range(self.n_steps - 1, -1, -1))

            for i in monit.iterate('Sample', len(time_steps) - 1):
                t = torch.tensor([time_steps[i]], device=self.device, dtype=torch.long)
                t_prev = torch.tensor([time_steps[i + 1]], device=self.device, dtype=torch.long) if i + 1 < len(time_steps) else torch.tensor([0], device=self.device)

                # ✅ Use Heun's method instead of basic DDIM
                x = self.diffusion.ddim_sample_heun(x, t, t_prev)

            # ✅ Final sanity check
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError("\u274C NaN or Inf detected in sampled tensor!")

            print(f"\u2705 Sampled tensor shape: {x.shape}, dtype: {x.dtype}")

            
#     def sample(self):
#         """
#         ### Sample images using DDIM Sampling
#         """
#         with torch.no_grad():
#             x = torch.randn([self.n_samples, self.image_channels, self.chunk_size], device=self.device)
#             
#             ## ✅ Sampling loop uses TSG-aware `ddim_sample()`
#             for t_ in monit.iterate('Sample', self.n_steps):
#                 t = torch.tensor(self.n_steps - t_ - 1, device=self.device, dtype=torch.long)
# 
#                 # print(f"\u2705 [Debug] Sampling step {t.cpu().item()}, Tensor Shape: {x.shape}")
# 
#                 if torch.any(t < 0) or torch.any(t >= self.n_steps):
#                     raise ValueError(f"\u274C Invalid time step index detected: {t}")
# 
#                 # ✅ Uses new TSG-based ddim_sample (time perturbation happens inside)
#                 x = self.diffusion.ddim_sample(x, t.unsqueeze(0))
# 
#             # ✅ Final sanity check
#             if torch.isnan(x).any() or torch.isinf(x).any():
#                 raise ValueError("\u274C NaN or Inf detected in sampled tensor!")
# 
#             print(f"\u2705 Sampled tensor shape: {x.shape}, dtype: {x.dtype}")
# 
#             # ✅ Save final sample
#             # tracker.save('sample', x.cpu())

    
    def save_model(self):
        """
        Save the entire model (parameters, weights, optimizer state, etc.)
        """
        model_state = {
            "model_state_dict": self.eps_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "configurations": self.__dict__,  # Save configurations as well
        }
        os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)  # Ensure directory exists
        torch.save(model_state, self.save_model_path)
        print(f"Model saved successfully at {self.save_model_path}")
    
    def load_model(self):
        """
        Load the entire model (parameters, weights, optimizer state, etc.)
        """
        if os.path.exists(self.save_model_path):
            try:
                checkpoint = torch.load(self.save_model_path, map_location=self.device , weights_only=False)  ##
                self.eps_model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print(f"Model loaded successfully from {self.save_model_path}")
            except Exception as e:
                print(f"Error loading the model: {e}")
        else:
            print(f"No checkpoint found at {self.save_model_path}")
   


    def train(self):
        """
        ### Train
        """
        # print(f"device selected {self.device}")

        # Iterate through the dataset
        for data in monit.iterate('Train', self.data_loader):
            # Increment global step
            tracker.add_global_step()
            # Move data to device
            # # print(f"data.dtype {data.dtype}")
            # print(f"device selected {self.device}")
            data = data.to(self.device)
            # print(f"data shape in train {data.shape}")
            # data = torch.stack(data).to(device)  # Convert list to tensor and move to device
            
            # Make the gradients zero
            self.optimizer.zero_grad()
            # Calculate loss
            # print(f"data shape in train {data.shape}")
            loss = self.diffusion.loss(data)
            # Compute gradients
            loss.backward()
            # Take an optimization step
            self.optimizer.step()
            # Track the loss
            # tracker.save('loss', loss)

    def run(self):
        """
        ### Training loop
        """
        # Determine which epochs should trigger image plotting (10 times in total)
        # plot_intervals = np.linspace(0, self.epochs - 1, num=10, dtype=int).tolist()
            
        # for  _ in monit.loop(self.epochs):
        for epoch_idx, _ in enumerate(monit.loop(self.epochs)):  # Explicit loop counter
            
            # Train the model
            self.train()
            
            # Only plot if the current epoch is in the precomputed plot intervals
            # if epoch_idx in plot_intervals:
            #     self.sample()  # Plot and save images
            
            # Sample some images
            self.sample()
            
            # New line in the console
            tracker.new_line()
            
            # Save the model **only every 5 epochs**
            if (epoch_idx + 1) % 10 == 0:  # Save at epoch 5, 10, 15, etc.
                self.save_model()
                print(f"Model saved at epoch {epoch_idx + 1}")  # Optional log statement
            
            # Save the model
            # Save model at every epoch
            # self.save_model()
            # experiment.save_checkpoint()


# 
# ## for 3 channels
# class TremorDataset(Dataset):
#     """
#     Custom dataset class for tremor data.
#     This dataset will use 'rsam', 'mf', 'hf', and 'dsar' as channels with 'time' as the sequence length.
#     """
# 
#     def __init__(self, file_path: str, start_time: str, end_time: str, chunk_size: int = 30):
#         # Read the data file
#         df = pd.read_csv(file_path, sep=",", parse_dates=['time'])
#         df['time'] = pd.to_datetime(df['time'])
# 
#         # Filter the data based on the user-defined time range
#         self.df_filtered = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
# 
#         # Extract values for all four channels
#         self.time_series_dt = self.df_filtered['time']
#         self.rsam_values = self.df_filtered['rsam'].values
#         self.mf_values = self.df_filtered['mf'].values
#         self.hf_values = self.df_filtered['hf'].values
#         # self.dsar_values = self.df_filtered['dsar'].values  # Include DSAR
# 
#         # Stack the four channels into a 2D array: (length, 4)
#         self.data = np.stack([self.rsam_values, self.mf_values, self.hf_values],  axis=-1) #self.dsar_values,
# 
#         # Define chunking properties
#         self.chunk_size = chunk_size
#         self.num_chunks = len(self.data) // self.chunk_size
# 
#         # If there's any remainder (less than a chunk_size), discard it
#         if len(self.data) % self.chunk_size != 0:
#             self.data = self.data[:self.num_chunks * self.chunk_size]
# 
#     def __len__(self):
#         """
#         Return the number of chunks in the dataset (based on chunk_size).
#         """
#         return self.num_chunks
#     
#     def __getitem__(self, index: int):
#         """
#         Get a sample (chunk of `chunk_size` time steps, with 4 channels).
#         """
#         # Get the start and end indices for the chunk
#         start_idx = index * self.chunk_size
#         end_idx = start_idx + self.chunk_size
# 
#         # Slice the data for the chunk
#         chunk_data = self.data[start_idx:end_idx]
# 
#         # Convert the chunk data into a tensor of shape (chunk_size, 4)
#         chunk_data_tensor = torch.tensor(chunk_data, dtype=torch.float32)
# 
#         # Reshape the tensor to (4, chunk_size) for model input
#         chunk_data_tensor = chunk_data_tensor.permute(1, 0)  # Shape becomes (4, chunk_size)
# 
#         # Get the time for the first sample in the chunk (used for labeling)
#         time = self.time_series_dt.iloc[start_idx]  
#         time_str = time.strftime('%Y-%m-%d %H:%M:%S')
# 
#         # Convert timestamp to tensor (if needed for your model)
#         time_tensor = torch.tensor([time.timestamp()], dtype=torch.float32)  
# 
#         return chunk_data_tensor

# i am loading the tremor dataset using following code. modify this code such that it loads the data for WADi dataset which has following format using head(2) command and it has 124 columns including the datetime.so i want the data to be 123 colums while datatime is just for reading the data. please note that we already have the stacked data in file so we dont need to do that we just need to load 123 columns data for model training.


""" For five channels"""
        
class TremorDataset(Dataset):
    """
    Custom dataset class for tremor data.
    This dataset uses 'rsam', 'mf', 'hf', 'dsar' and 'ssam' as channels,
    with 'time' defining the sequence length.
    """

    def __init__(self, file_path: str, start_time: str, end_time: str, chunk_size: int = 30):
        # Read the data file
        df = pd.read_csv(file_path, sep=",", parse_dates=['time'])
        df['time'] = pd.to_datetime(df['time'])

        # Filter the data based on the user-defined time range
        mask = (df['time'] >= pd.to_datetime(start_time)) & (df['time'] <= pd.to_datetime(end_time))
        self.df_filtered = df.loc[mask].reset_index(drop=True)

        # Extract the time series for labeling
        self.time_series_dt = self.df_filtered['time']

        # Extract values for all five channels
        self.rsam_values = self.df_filtered['rsam'].values
        self.mf_values   = self.df_filtered['mf'].values
        self.hf_values   = self.df_filtered['hf'].values
        self.dsar_values = self.df_filtered['dsar'].values
        self.ssam_values = self.df_filtered['ssam'].values

        # Stack the five channels into a 2D array: (length, 5)
        self.data = np.stack([
            self.rsam_values,
            self.mf_values,
            self.hf_values,
            self.dsar_values,
            self.ssam_values
        ], axis=-1)

        # Define chunking properties
        self.chunk_size = chunk_size
        self.num_chunks = len(self.data) // self.chunk_size

        # If there's any remainder (< chunk_size), discard it
        total_used = self.num_chunks * self.chunk_size
        if total_used != len(self.data):
            self.data = self.data[:total_used]
            self.time_series_dt = self.time_series_dt.iloc[:total_used]

    def __len__(self):
        """
        Number of chunks in the dataset.
        """
        return self.num_chunks

    def __getitem__(self, index: int):
        """
        Return a tuple (chunk_tensor, time_tensor):
          - chunk_tensor: FloatTensor of shape (5, chunk_size)
          - time_tensor:  FloatTensor of shape (1,) with the timestamp of the first sample
        """
        start_idx = index * self.chunk_size
        end_idx   = start_idx + self.chunk_size

        # Slice out the chunk: shape (chunk_size, 5)
        chunk = self.data[start_idx:end_idx]

        # Convert to FloatTensor and permute to (5, chunk_size)
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).permute(1, 0)

        # First timestamp in the chunk
        t0 = self.time_series_dt.iloc[start_idx]
        time_tensor = torch.tensor([t0.timestamp()], dtype=torch.float32)

        return chunk_tensor #, time_tensor


@option(Configs.dataset, 'Tremor')
def tremor_dataset(c: Configs):
    """
    Create Tremor dataset
    """
    file_path = c.dataset_path
    #"./data_external/whakaari-master/data/clean_tremor_data.csv"
    # file_path = "./data_external/whakaari-master/data/tremor_data_norm.dat"
    start_time = "2007-11-01"
    end_time = "2025-02-09"
    chunk_size= c.chunk_size
    
    return TremorDataset(file_path, start_time, end_time,chunk_size)

## for WaDi Dataset
## 

class WADIDataset(Dataset):
    """
    Custom Dataset class for the WADI dataset.

    The stacked WADI file is expected to have 124 columns:
      - The first column ("Datetime") contains timestamps.
      - The remaining 123 columns contain sensor readings.
    
    For model training, the datetime column is only used for filtering
    and is then dropped so that the training data contain exactly 123 columns.
    """
    def __init__(self, file_path: str, start_time: str, end_time: str, chunk_size: int = 30):
        # Read the CSV file, parsing the "Datetime" column.
        # If your CSV file was saved with an extra index column (e.g. "Unnamed: 0"),
        # remove it.
        df = pd.read_csv(file_path, sep=",", parse_dates=['Datetime'])
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        
        # Ensure that "Datetime" is in datetime format.
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Filter rows based on the specified time range.
        self.df_filtered = df[(df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)]
        
        # Optionally, save the datetime column for reference.
        self.time_series_dt = self.df_filtered['Datetime']
        
        # Drop the "Datetime" column so that only sensor data remain.
        self.data = self.df_filtered.drop(columns=['Datetime']).values
        
        # Check that we now have exactly 123 sensor columns.
        if self.data.shape[1] != 123:
            raise ValueError(f"Expected 123 sensor columns, but got {self.data.shape[1]} columns.")
        
        # Define chunking properties: create chunks of consecutive rows.
        self.chunk_size = chunk_size
        self.num_chunks = len(self.data) // self.chunk_size
        
        # Discard remainder rows that do not make up a full chunk.
        if len(self.data) % self.chunk_size != 0:
            self.data = self.data[:self.num_chunks * self.chunk_size]

    def __len__(self):
        # Return the number of chunks.
        return self.num_chunks

    def __getitem__(self, index: int):
        """
        Return a chunk (sample) of the data as a torch tensor.
        
        The returned tensor is of shape (123, chunk_size), with sensor channels
        as the first dimension.
        """
        start_idx = index * self.chunk_size
        end_idx = start_idx + self.chunk_size
        
        # Slice the data to get the current chunk (shape: (chunk_size, 123))
        chunk_data = self.data[start_idx:end_idx]
        
        # Convert to a tensor and then permute to shape (123, chunk_size)
        chunk_data_tensor = torch.tensor(chunk_data, dtype=torch.float32)
        chunk_data_tensor = chunk_data_tensor.permute(1, 0)
        return chunk_data_tensor

# Example usage function (adapt as needed for your configuration)
@option(Configs.dataset, 'WADI')
def wadi_dataset(c: Configs):
    """
    Utility function to create a WADI dataset instance.
    
    The start_time and end_time should be strings that pandas can parse as timestamps,
    for example, "2017-09-25 18:00:00".
    """
    file_path = c.dataset_path  # e.g. "/path/to/WADI_stacked.csv"
    start_time = "2017-09-25 18:00:00"
    end_time   = "2018-09-25 23:59:59"
    chunk_size = c.chunk_size
    return WADIDataset(file_path, start_time, end_time, chunk_size)


class SWATDataset(Dataset):
    """
    Dataset for SWAT data saved in .npy with 51 features per sample.
    Uses integer row indices as pseudo‐time for tracking.
    """

    def __init__(self, file_path: str, chunk_size: int = 50):
        # Load the .npy file
        arr = np.load(file_path)
        if arr.ndim != 2 or arr.shape[1] != 51:
            raise ValueError(f"Expected data of shape (n_samples, 51), got {arr.shape}")

        # Ensure float32 for model input
        self.data = arr.astype(np.float32)  # shape: (n_samples, 51)

        # Row indices as time_index
        self.time_index = np.arange(len(self.data))

        # Define chunking
        self.chunk_size = chunk_size
        self.num_chunks = len(self.data) // self.chunk_size

        # Truncate to an exact multiple of chunk_size
        total = self.num_chunks * self.chunk_size
        self.data = self.data[:total]
        self.time_index = self.time_index[:total]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index: int):
        start = index * self.chunk_size
        end   = start + self.chunk_size

        # Slice out the chunk
        chunk = self.data[start:end]               # (chunk_size, 51)
        # Convert to tensor and permute to (features, timesteps)
        chunk_tensor = torch.from_numpy(chunk).permute(1, 0)  # (51, chunk_size)

        # Optional: return time‐indices as well
        time_idx = self.time_index[start:end]      # (chunk_size,)

        return chunk_tensor#, torch.from_numpy(time_idx.astype(np.int64))

# If you use a registry or config helper:
@option(Configs.dataset, 'SWAT')
def swat_dataset(c: Configs):
    return SWATDataset(file_path=c.dataset_path, chunk_size=c.chunk_size)



class SyntheticDataset(Dataset):
    """
    Dataset for synthetic data with 5 features and no timestamp column.
    Uses row indices as pseudo time for tracking.
    """

    def __init__(self, file_path: str, chunk_size: int = 30):
        # Load the CSV file (no header expected, or adjust if needed)
        df = pd.read_csv(file_path, sep=",")
        
        # Ensure it's float32 for training
        self.data = df.values.astype(np.float32)  # shape: (n_samples, 5)

        # Row indices as time_index
        self.time_index = np.arange(len(self.data))

        # Define chunking properties
        self.chunk_size = chunk_size
        self.num_chunks = len(self.data) // self.chunk_size

        # Truncate to multiple of chunk_size
        self.data = self.data[:self.num_chunks * self.chunk_size]
        self.time_index = self.time_index[:self.num_chunks * self.chunk_size]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, index: int):
        start_idx = index * self.chunk_size
        end_idx = start_idx + self.chunk_size

        # Get the chunk of data and convert to tensor
        chunk_data = self.data[start_idx:end_idx]  # shape: (chunk_size, 5)
        chunk_tensor = torch.tensor(chunk_data).permute(1, 0)  # (5, chunk_size)
        # print(f"data shape is {chunk_tensor.shape}")
        # Time index for plotting/reference
        chunk_time_index = self.time_index[start_idx:end_idx]

        return chunk_tensor#, chunk_time_index

@option(Configs.dataset, 'Synthetic')
def synthetic_dataset(c: Configs):
    return SyntheticDataset(file_path=c.dataset_path, chunk_size=c.chunk_size)


class YahooTrainDataset(Dataset):
    """
    Training‐only loader for Yahoo Sub 5.
    Drops timestamp & ground_truth, stacks the 5 feature columns,
    and chunks into (chunk_size) windows.
    """
    def __init__(self, file_path: str, chunk_size: int = 30):
        # 1) Load the CSV
        df = pd.read_csv(file_path, sep=",")
        # 2) Drop non‐feature columns
        for c in ("Index", "timestamp", "ground_truth"):
            if c in df.columns:
                df = df.drop(columns=[c])
        # 3) Convert to float32 array
        self.data = df.values.astype(np.float32)  # (n_samples, 5)
        # 4) Pseudo‐time (if you ever need it)
        self.time_index = np.arange(len(self.data))

        # 5) Chunking
        self.chunk_size = chunk_size
        self.num_chunks = len(self.data) // chunk_size
        # truncate
        total = self.num_chunks * chunk_size
        self.data = self.data[:total]
        self.time_index = self.time_index[:total]

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx: int):
        s = idx * self.chunk_size
        e = s + self.chunk_size
        block = self.data[s:e]  # (chunk_size, 5)
        # print(f"shape of data {block.shape}, {block[0]}")
        # to tensor (5, chunk_size)
        return torch.tensor(block, dtype=torch.float32).permute(1, 0)

# -----------------------------------------------------------------------------
# register for your Configs.dataset machinery
# -----------------------------------------------------------------------------

@option(Configs.dataset, 'Yahoo')
def yahoo_train_dataset(c: Configs):
    """
    Use YahooSub5 training data (no labels).
    """
    return YahooTrainDataset(file_path=c.dataset_path,
                             chunk_size=c.chunk_size)


def main():
    # Create experiment
    experiment.create(name='diffuse', writers={'screen', 'labml'})

    # Create configurations
    configs = Configs()

    # Set configurations. You can override the defaults by passing the values in the dictionary.
    # experiment.configs(configs, {
    #     'dataset': 'CelebA',  # 'MNIST'
    #     'image_channels': 3,  # 1,
    #     'epochs': 100,  # 5,
    # })
    
    if configs.data_name == 'WADI':
        print("the dataset selected is WADI")
        experiment.configs(configs, {
            'dataset': 'WADI',
            'image_channels': 123,
            'epochs': 100,
            "dataset_path":"./data_external/time_series_datasets/WaDi/WADI.A1_9 Oct 2017/df_normal.csv"
        })

    elif configs.data_name == 'Tremor':
        print("the dataset selected is Tremor")
        experiment.configs(configs, {
            'dataset': 'Tremor',
            'image_channels': 5,
            'epochs': 100,
            "dataset_path": "./cfg/data/raw/Whakaari_clean_tremor_data_mydata_ssam_norm.csv"
        })
         
    elif configs.data_name == 'Yahoo':
        print("the dataset selected is Yahoo")
        experiment.configs(configs, {
            'dataset': 'Yahoo',
            'image_channels': 5,
            'epochs': 100,
            "dataset_path":"./cfg/data/processed/yahoo/learningData_yahoo_train.csv"
        })
    elif configs.data_name == 'Synthetic':
        print("the dataset selected is Synthetic")
        experiment.configs(configs, {
            'dataset': 'Synthetic',
            'image_channels': 5,
            'epochs': 100,
            "dataset_path": "./cfg/data/processed/synth_data/pattern_seasonal/train.csv"
        })

        

    # Initialize
    configs.init()
    print(f"configs.dataset{configs.dataset}")
    print(f"configs.dataset{configs.guidance_scale}")
    # Set models for saving and loading
    experiment.add_pytorch_models({'eps_model': configs.eps_model})

    # Start and run the training loop
    with experiment.start():
        configs.run()


#
if __name__ == '__main__':
    main()
