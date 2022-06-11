"""Library for load operation. .
"""
import re


def extract_obsid(string: str, digit: int):
    obsid = re.findall(f"[0-9]{{{digit}}}", string)[0]
    return obsid
