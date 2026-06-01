"""PRISM report generation modules."""
from .coverage import generate_coverage_report, CoverageReport
from .novel_summary import generate_novel_summary, NovelSummaryReport

__all__ = [
    'generate_coverage_report', 'CoverageReport',
    'generate_novel_summary', 'NovelSummaryReport',
]
