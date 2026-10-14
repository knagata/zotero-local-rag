#!/usr/bin/env python3
"""Apply one explicitly reviewed draft; the admin job runs all audits afterwards."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
from src.env_utils import load_dotenv_native
load_dotenv_native(ROOT)
from citation_graph.debug_apply import apply_review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attachment',required=True)
    parser.add_argument('--revision',required=True,type=int)
    args=parser.parse_args()
    apply_review(ROOT,args.attachment,args.revision)
    print('[修正反映] 本文・文書構造を更新しました。全DB監査を実行してください。',flush=True)


if __name__=='__main__':
    main()
