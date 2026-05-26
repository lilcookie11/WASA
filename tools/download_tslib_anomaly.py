import argparse
import shutil
import urllib.request
from pathlib import Path


FILES = {
    "PSM": ["train.csv", "test.csv", "test_label.csv"],
    "MSL": ["MSL_train.npy", "MSL_test.npy", "MSL_test_label.npy"],
    "SMD": ["SMD_train.npy", "SMD_test.npy", "SMD_test_label.npy"],
}


def download_file(repo_id, dataset, filename, output_root, backend):
    repo_path = f"{dataset}/{filename}"
    target_dir = output_root / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    if backend == "hub":
        from huggingface_hub import hf_hub_download

        source = Path(hf_hub_download(repo_id=repo_id, filename=repo_path, repo_type="dataset"))
        shutil.copy2(source, target)
    else:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{repo_path}"
        tmp = target.with_suffix(target.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
    print(f"{repo_path} -> {target} ({target.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="thuml/Time-Series-Library")
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--datasets", nargs="+", default=["PSM", "MSL", "SMD"], choices=sorted(FILES))
    parser.add_argument("--backend", type=str, default="direct", choices=["direct", "hub"])
    args = parser.parse_args()

    for dataset in args.datasets:
        for filename in FILES[dataset]:
            download_file(args.repo_id, dataset, filename, args.output, args.backend)


if __name__ == "__main__":
    main()
