"""Test RoMa v2 import and availability.

This test verifies that the romav2 package is installed and the RoMaV2 class
is available. We do not instantiate the full model (which downloads weights and
is slow) -- just verify the constructor is callable.

Note: The romav2 package exports the main RoMaV2 class directly. The class
constructor accepts an optional Cfg parameter and can be called with no arguments
to use default settings (though weight loading would happen on instantiation).
"""


def test_romav2_import():
    """Test that romav2 package can be imported."""
    import romav2

    assert romav2 is not None


def test_romav2_class_available():
    """Test that RoMaV2 class is accessible and callable."""
    from romav2 import RoMaV2

    assert callable(RoMaV2), "RoMaV2 should be callable"


def test_romav2_configure_logger():
    """Test that configure_logger function is available."""
    from romav2 import configure_logger

    assert callable(configure_logger), "configure_logger should be callable"
