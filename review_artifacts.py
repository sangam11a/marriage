"""
Root runner script for 13-point Q1 Reviewer Audit.
"""
import sys
import os

from marriage.review_artifacts import run_reviewer_audit

if __name__ == "__main__":
    success = run_reviewer_audit()
    sys.exit(0 if success else 1)
