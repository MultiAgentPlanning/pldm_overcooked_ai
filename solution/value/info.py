#!/usr/bin/env python3
import argparse
from solution.pldm.data_processor import get_overcooked_dataloaders

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to CSV")
    p.add_argument("--enc",  choices=["grid","vector"], default="grid")
    p.add_argument("--bs",   type=int, default=1)
    args = p.parse_args()

    tr, _, _ = get_overcooked_dataloaders(args.data, args.enc, batch_size=args.bs)
    print("Inspecting first 10 actions in train dataset:")
    for idx, (s, a, ns, r) in enumerate(tr.dataset):
        print(f"  [{idx}] action tensor shape: {tuple(a.size())}, dtype: {a.dtype}, values: {a}")
        if idx >= 9:
            break

if __name__ == "__main__":
    main()
