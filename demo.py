"""
Root runner script for 21-Card Nepali Marriage AI Pipeline Demo.
"""
import sys
import os

from marriage.demo import run_pipeline_demo

if __name__ == "__main__":
    run_pipeline_demo(num_deals=8)
