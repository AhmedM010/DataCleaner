import pandas as pd
from pathlib import Path
import numpy as np
import re

# CONSTANTS
LABEL_TO_ID =  {"A": 0,
                "A_PRIME": 1,
                "B": 2,
                "C": 3}

WINDOW_SIZE = 128
FEATURE_COUNT = 9
TRAIN_ROUND = 1
TEST_ROUND = 2
rate_hz = 50          # contract: Module 2 delivers 50 Hz
min_duration = 275    # 5 min chunk, 10s dropped at each edge, with 5 buffer for safety

sensor_columns = ['acc_x','acc_y','acc_z',
                'gyr_x','gyr_y','gyr_z',
                'grav_x','grav_y','grav_z']


def load_sensor_data():
    
    data_folder = Path("datas")
    recordings={}
    acc_rename = {"x": "acc_x", "y": "acc_y", "z": "acc_z"}
    gyr_rename = {"x": "gyr_x", "y": "gyr_y", "z": "gyr_z"}
    grav_rename = {"x": "grav_x", "y": "grav_y", "z": "grav_z"}


    for folder in data_folder.iterdir():
        
        # skip anything that isn't a folder
        if not folder.is_dir():
            continue        
        # try to read the files. if any are missing, skip this folder.
        try:
            acc  = pd.read_csv(folder / "Accelerometer.csv")
            gyr  = pd.read_csv(folder / "Gyroscope.csv")
            grav = pd.read_csv(folder / "Gravity.csv")
        except FileNotFoundError:
            print(f"Skipping {folder.name}: missing sensor file.")
            continue

        acc  = acc.rename(columns=acc_rename)
        gyr  = gyr.rename(columns=gyr_rename)
        grav = grav.rename(columns=grav_rename)
        
        recordings[folder.name] = (acc, gyr, grav)
    return recordings

def merge_sensor_data(acc, gyr, grav):

    
    acc_time_diff     = acc["time"].diff() / 1_000_000_000
    acc_average_diff  = acc_time_diff.mean()
    acc_sample_rate   = 1 / acc_average_diff 
    
    gyr_time_diff     = gyr["time"].diff() / 1_000_000_000
    gyr_average_diff  = gyr_time_diff.mean()
    gyr_sample_rate   = 1 / gyr_average_diff
    
    grav_time_diff    = grav["time"].diff() / 1_000_000_000
    grav_average_diff = grav_time_diff.mean()
    grav_sample_rate  = 1 / grav_average_diff

    if abs(acc_sample_rate - rate_hz) > 0.5:
        print("Accelerometer needs resampling")
    else:
        print("Accelerometer is already 50 Hz")

    if abs(gyr_sample_rate - rate_hz) > 0.5:
        print("Gyroscope needs resampling")
    else:
        print("Gyroscope is already 50 Hz")

    if abs(grav_sample_rate - rate_hz) > 0.5:
        print("Gravity needs resampling")
    else:
        print("Gravity is already 50 Hz")

    acc["time"]  = pd.to_datetime(acc["time"],  unit="ns")
    gyr["time"]  = pd.to_datetime(gyr["time"],  unit="ns")
    grav["time"] = pd.to_datetime(grav["time"], unit="ns")

    acc  = acc.set_index("time")
    gyr  = gyr.set_index("time")
    grav = grav.set_index("time")

    start_time = acc.index[0]


    acc  = acc.resample("20ms", origin=start_time).mean() 
    gyr  = gyr.resample("20ms", origin=start_time).mean()
    grav = grav.resample("20ms", origin=start_time).mean()

    print(acc_sample_rate, gyr_sample_rate, grav_sample_rate)


    gyr = gyr.drop(columns=["seconds_elapsed"])
    grav = grav.drop(columns=["seconds_elapsed"])

    merged = pd.merge(
        acc, gyr, left_index=True,
        right_index=True, how="inner")

    yallahmerged = pd.merge(
        merged, grav,left_index=True,
        right_index=True,how="inner")

    yallahmerged = yallahmerged.dropna(
        subset=sensor_columns,
        how="all")

    qalasmerged = yallahmerged.reset_index()
    return qalasmerged

def clean_table(df):
    """
    Input:  one merged df (Module 2 output)
    Output: one cleaned copy; input is not modified
    """

    # 1. make sure the df actually exists and has something
    assert not df.empty, "no data at all..."

    # 2. sort by time
    df = df.sort_values('seconds_elapsed').reset_index(drop=True)

    # 3. drop first 10s and last 10s
    t_min = df['seconds_elapsed'].min()
    t_max = df['seconds_elapsed'].max()
    df = df[(df['seconds_elapsed'] > t_min + 10) &
            (df['seconds_elapsed'] < t_max - 10)]
    duration = df['seconds_elapsed'].max() - df['seconds_elapsed'].min()

    # 4. drop NaN in sensor columns only
    df = df.dropna(subset=sensor_columns)

    # 5. checks for usability
    assert not df.empty, "no rows left after cleaning"
    expected_rows = rate_hz * duration * 0.9   # maximum of 10% data dropout
    assert duration >= min_duration, f"duration too short: {duration}"
    assert len(df) >= expected_rows, f"rows too few: {len(df)} < {expected_rows}"

    # 6. return for mod 4
    return df.reset_index(drop=True)

def window_cutting (df):
    # 1. set the variables
    values = df[sensor_columns].values
    window_size = WINDOW_SIZE
    step = 64
    windows = []

    # 2. cutting df into 128 sample windows at 64 step
    for sample in range(0, len(values) - window_size + 1, step):
        windows.append(values[sample:sample + window_size])

    # 3. check if the windows were made
    assert len(windows) > 0, "no windows were made: not enough samples"

    # 4. convert in array
    windows = np.array(windows)

    # 5. check the shape of the windows is (n, 128, 9)
    assert windows.shape[1:] == (WINDOW_SIZE, FEATURE_COUNT), f"bad shape: {windows.shape}"  

    # 6. return
    return windows

def label_and_stack(file_windows):
    """
    MODULE 5 - LABEL

    Input:
        file_windows = list of (filename, windows) pairs
        filename = CSV filename
        windows = NumPy array from Module 4
                    Shape: (n_windows, 128, 9)

    Output:
        X = combined sensor windows
        y = numerical labels for each window
        rounds = round number per window (1 or 2)
    """

    all_windows = []
    all_labels = []
    all_rounds = []

    for filename, windows in file_windows:
        file_stem = Path(filename).stem
        match = re.match(
            r"^(A(?:[_ -]?PRIME)?|B|C)[_\- ]?r(\d+)",
            file_stem,
            re.IGNORECASE
        )

        if match is None:
            raise ValueError(
                f"Cannot identify state/round from filename: {filename}"
            )
            
        state = match.group(1)
        state = re.sub(r"[_ -]", "", state).upper()
        if state == "APRIME":
            state = "A_PRIME"
        
        round_num = int(match.group(2))
        label = LABEL_TO_ID[state]
        
        windows = np.asarray(windows)
        if windows.ndim != 3 or windows.shape[1:] != (
            WINDOW_SIZE,
            FEATURE_COUNT
        ):
            raise ValueError(
                f"Invalid window shape for {filename}: {windows.shape}. "
                f"Expected (n_windows, 128, 9)."
            )

        if not np.issubdtype(windows.dtype, np.number):
            raise TypeError(
                f"Windows must contain numeric data: {filename}"
            )

        if windows.shape[0] == 0:
            continue
        
        all_windows.append(windows)
        all_labels.append(np.full(windows.shape[0], label,     dtype=np.int64))
        all_rounds.append(np.full(windows.shape[0], round_num, dtype=np.int64))
        

    if not all_windows:
        raise ValueError("No valid windows were provided.")

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    rounds = np.concatenate(all_rounds, axis=0)
    return X, y, rounds   

"""Module 6 - Split.

Splits Module 5's data by day (round 1 = train, round 2 = test)
and saves each set as a .npy file.
"""

def split_and_save(X, y, rounds, out_dir="data/split"):
    X, y, rounds = np.asarray(X), np.asarray(y), np.asarray(rounds)
    _validate(X, y, rounds)

    # True/False mask for each window: is it from round 1 or round 2?
    train = rounds == TRAIN_ROUND
    test = rounds == TEST_ROUND

    # The masks pick out only the matching windows
    sets = {
        "X_train": X[train],
        "y_train": y[train],
        "X_test": X[test],
        "y_test": y[test],
    }
    _check_labels(sets["y_train"], sets["y_test"])

    # Create the output folder if it doesn't exist yet
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, arr in sets.items():
        path = out / f"{name}.npy"
        np.save(path, arr, allow_pickle=False)  # no pickle = safer to load
        paths[name] = str(path)

    print(f"train: {train.sum()} windows | test: {test.sum()} windows")
    return paths

def _validate(X, y, rounds):
    # Every window needs exactly one label and one round
    if not (len(X) == len(y) == len(rounds)):
        raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, rounds={len(rounds)}")
    if len(X) == 0:
        raise ValueError("Empty dataset from Module 5.")

    # Only rounds 1 and 2 are allowed, and both must be there
    unknown = set(np.unique(rounds)) - {TRAIN_ROUND, TEST_ROUND}
    if unknown:
        raise ValueError(f"Unexpected round values: {sorted(unknown)}")
    for r in (TRAIN_ROUND, TEST_ROUND):
        if not np.any(rounds == r):
            raise ValueError(f"No windows for round {r}.")

def _check_labels(y_train, y_test):
    train_labels, test_labels = set(np.unique(y_train)), set(np.unique(y_test))

    # Model can't be tested on an activity it never trained on
    if test_labels - train_labels:
        raise ValueError(f"Test has unseen labels: {sorted(test_labels - train_labels)}")

    # Not fatal, but that activity won't be tested
    if train_labels - test_labels:
        print(f"Warning: no test data for: {sorted(train_labels - test_labels)}")

def DataCleaner():
    # Module 1 — load all recordings
    recordings = load_sensor_data()

    # Modules 2-4 — run per recording
    file_windows = []
    for folder_name, (acc, gyr, grav) in recordings.items():
        print(f"\n>>> Reading folder: {folder_name}")
        try:
            merged  = merge_sensor_data(acc, gyr, grav)
            cleaned = clean_table(merged)
            windows = window_cutting(cleaned)
            file_windows.append((folder_name, windows))
        except AssertionError as e:
            print(f"Skipping {folder_name}: {e}")
            continue

    # Module 5 — label and stack
    X, y, rounds = label_and_stack(file_windows)

    # Module 6 — split by round
    split_and_save(X, y, rounds)

    return X, y, rounds

if __name__ == "__main__":
    DataCleaner()
