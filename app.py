"""Default Streamlit entry point for the VulnScope v2 meeting demo."""

from pathlib import Path
import runpy


# Streamlit reruns this entry point after every widget interaction. ``run_path``
# deliberately executes the v2 script again instead of relying on Python's
# import cache, so uploads and button clicks render their updated results.
runpy.run_path(
    str(Path(__file__).with_name("app_v2.py")),
    run_name="__main__",
)
