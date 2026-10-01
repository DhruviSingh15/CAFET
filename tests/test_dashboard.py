import pytest
from streamlit.testing.v1 import AppTest

import os
def test_dashboard_loads():
    at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "../dashboard.py"), default_timeout=30).run()
    assert not at.exception
    
    # Simulate selecting diabetes
    at.selectbox[0].set_value("diabetes")
    # Simulate clicking run button
    at.button[0].click().run()
    
    assert not at.exception
    
    # Check if results rendered (e.g. success or info elements are present)
    assert len(at.success) > 0 or len(at.info) > 0 or len(at.warning) > 0
    assert "7. Learn: Historical Experience" in [h.value for h in at.header]
