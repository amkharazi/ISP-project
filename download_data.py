import os
import requests
import argparse

def download_miller_dataset():
    fname = 'motor_imagery.npz'
    url = "https://osf.io/ksqv8/download"

    if os.path.isfile(fname):
        print(f"Dataset '{fname}' already exists. Skipping download.")
    else:
        try:
            print(f"Downloading dataset from {url}...")
            r = requests.get(url)
            r.raise_for_status()
        except requests.ConnectionError:
            print("!!! Failed to download data: Connection error !!!")
        except requests.HTTPError as http_err:
            print(f"!!! HTTP error occurred: {http_err} !!!")
        else:
            with open(fname, "wb") as fid:
                fid.write(r.content)
            print(f"Dataset '{fname}' downloaded successfully.")

def main():
    parser = argparse.ArgumentParser(description="Dataset downloader script.")
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='miller', 
        choices=['miller'], 
        help="Specify the dataset to download (default: 'miller')."
    )
    
    args = parser.parse_args()

    if args.dataset.lower() == 'miller':
        download_miller_dataset()
    else:
        print(f"Dataset '{args.dataset}' is not recognized. Available datasets: 'miller'.")

if __name__ == "__main__":
    main()